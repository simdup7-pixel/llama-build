#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "llama.h"

#define TAG "LlamaBridge"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model*         g_model = nullptr;
static llama_context*       g_ctx   = nullptr;
static llama_sampler*       g_smpl  = nullptr;
static const llama_vocab*   g_vocab = nullptr;
static const char*          g_tmpl  = nullptr;
static std::vector<llama_chat_message> g_msgs;
static std::vector<char>    g_fmt(8192);
static int                  g_prev_len = 0;
static int                  g_n_past = 0;
static int                  g_n_ctx = 2048;
static const int            g_n_batch = 512;
static std::atomic<bool>    g_stop{false};

static void free_msgs() {
    for (auto& m : g_msgs) free((void*)m.content);
    g_msgs.clear();
}

static void reset_state() {
    free_msgs();
    g_prev_len = 0;
    g_n_past = 0;
    if (g_ctx) llama_memory_clear(llama_get_memory(g_ctx), true);
    if (g_smpl) llama_sampler_reset(g_smpl);
}

static int apply(bool add_ass) {
    auto run = [&](const char* t) {
        int n = llama_chat_apply_template(t, g_msgs.data(), g_msgs.size(), add_ass,
                                          g_fmt.data(), (int)g_fmt.size());
        if (n > (int)g_fmt.size()) {
            g_fmt.resize(n);
            n = llama_chat_apply_template(t, g_msgs.data(), g_msgs.size(), add_ass,
                                          g_fmt.data(), (int)g_fmt.size());
        }
        return n;
    };
    int n = g_tmpl ? run(g_tmpl) : -1;
    if (n < 0) n = run("chatml");
    return n;
}

static size_t utf8_complete_len(const std::string& s) {
    size_t n = s.size();
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        i--; back++;
        unsigned char c = (unsigned char)s[i];
        if ((c & 0xC0) != 0x80) {
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            return need > back ? i : n;
        }
    }
    return n;
}

static void emit(JNIEnv* env, jobject self, jmethodID mid, const std::string& s) {
    if (s.empty()) return;
    jbyteArray arr = env->NewByteArray((jsize)s.size());
    env->SetByteArrayRegion(arr, 0, (jsize)s.size(), (const jbyte*)s.data());
    env->CallVoidMethod(self, mid, arr);
    env->DeleteLocalRef(arr);
}

static bool decode_all(std::vector<llama_token>& t) {
    for (size_t i = 0; i < t.size(); i += g_n_batch) {
        if (g_stop) return false;
        int n = (int)std::min<size_t>(g_n_batch, t.size() - i);
        llama_batch b = llama_batch_get_one(t.data() + i, n);
        if (llama_decode(g_ctx, b) != 0) return false;
        g_n_past += n;
    }
    return true;
}

static void n_free(JNIEnv*, jobject) {
    free_msgs();
    if (g_smpl)  { llama_sampler_free(g_smpl);  g_smpl = nullptr; }
    if (g_ctx)   { llama_free(g_ctx);           g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model);   g_model = nullptr; }
}

static jboolean n_load(JNIEnv* env, jobject, jstring jpath, jint nCtx, jint nThreads) {
    n_free(env, nullptr);
    const char* path = env->GetStringUTFChars(jpath, nullptr);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) { LOGE("load model gagal"); return JNI_FALSE; }

    g_vocab = llama_model_get_vocab(g_model);
    g_tmpl  = llama_model_chat_template(g_model, nullptr);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nCtx;
    cp.n_batch = g_n_batch;
    cp.n_threads = nThreads;
    cp.n_threads_batch = nThreads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { LOGE("init context gagal"); return JNI_FALSE; }
    g_n_ctx = (int)llama_n_ctx(g_ctx);

    g_smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_p(0.8f, 1));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    reset_state();
    return JNI_TRUE;
}

static void n_generate(JNIEnv* env, jobject self, jstring jtext) {
    jmethodID mid = env->GetMethodID(env->GetObjectClass(self), "emit", "([B)V");
    g_stop = false;

    const char* text = env->GetStringUTFChars(jtext, nullptr);
    g_msgs.push_back({"user", strdup(text)});
    env->ReleaseStringUTFChars(jtext, text);

    int new_len = apply(true);
    if (new_len < 0 || new_len < g_prev_len) {
        free((void*)g_msgs.back().content); g_msgs.pop_back();
        emit(env, self, mid, "[Error: format chat gagal]");
        return;
    }
    std::string prompt(g_fmt.data() + g_prev_len, new_len - g_prev_len);

    bool first = (g_n_past == 0);
    int n = -llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, first, true);
    std::vector<llama_token> toks(n);
    if (llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), toks.data(), n, first, true) < 0) {
        free((void*)g_msgs.back().content); g_msgs.pop_back();
        emit(env, self, mid, "[Error: tokenize gagal]");
        return;
    }

    if (g_n_past + n + 8 > g_n_ctx) {
        free((void*)g_msgs.back().content); g_msgs.pop_back();
        emit(env, self, mid, "[Konteks penuh. Panggil reset() untuk mulai chat baru]");
        return;
    }

    if (!decode_all(toks)) {
        if (!g_stop) emit(env, self, mid, "[Error: decode gagal]");
        reset_state();
        return;
    }

    std::string response, pending;
    while (!g_stop && g_n_past < g_n_ctx - 1) {
        llama_token tok = llama_sampler_sample(g_smpl, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;

        char buf[256];
        int len = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, false);
        if (len < 0) break;
        pending.append(buf, len);
        response.append(buf, len);

        size_t ok = utf8_complete_len(pending);
        if (ok > 0) {
            emit(env, self, mid, pending.substr(0, ok));
            pending.erase(0, ok);
        }

        llama_batch b = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, b) != 0) break;
        g_n_past++;
    }

    g_msgs.push_back({"assistant", strdup(response.c_str())});
    int len = apply(false);
    g_prev_len = len < 0 ? 0 : len;
}

static void n_stop(JNIEnv*, jobject)  { g_stop = true; }
static void n_reset(JNIEnv*, jobject) { reset_state(); }

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass c = env->FindClass(BRIDGE_CLASS);
    if (!c) { LOGE("class %s tidak ditemukan (package name salah?)", BRIDGE_CLASS); return JNI_ERR; }
    static const JNINativeMethod m[] = {
        {"nativeLoad",     "(Ljava/lang/String;II)Z", (void*)n_load},
        {"nativeGenerate", "(Ljava/lang/String;)V",   (void*)n_generate},
        {"nativeStop",     "()V",                     (void*)n_stop},
        {"nativeReset",    "()V",                     (void*)n_reset},
        {"nativeFree",     "()V",                     (void*)n_free},
    };
    env->RegisterNatives(c, m, 5);
    return JNI_VERSION_1_6;
}
