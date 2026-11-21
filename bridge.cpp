#include <jni.h>
#include <string>
#include <vector>
#include "llama.h"

static llama_model* model = nullptr;
static llama_context* ctx = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_llm_chat_Bridge_load(JNIEnv* env, jclass, jstring path) {
  const char* p = env->GetStringUTFChars(path, 0);
  llama_backend_init();
  model = llama_model_load_from_file(p, llama_model_default_params());
  env->ReleaseStringUTFChars(path, p);
  if (!model) return JNI_FALSE;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 2048; cp.n_batch = 512;
  cp.n_threads = 4; cp.n_threads_batch = 4;
  ctx = llama_init_from_model(model, cp);
  return ctx ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_llm_chat_Bridge_generate(JNIEnv* env, jclass, jstring prompt) {
  std::string out;
  if (!ctx) { out = "Model belum dimuat"; }
  else {
    const char* c = env->GetStringUTFChars(prompt, 0);
    std::string text = std::string("<|im_start|>user\n") + c +
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    env->ReleaseStringUTFChars(prompt, c);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_memory_clear(llama_get_memory(ctx), true);
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), n, true, true);
    llama_sampler* s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(s, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(s, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(s, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    llama_token t;
    llama_batch b = llama_batch_get_one(toks.data(), toks.size());
    for (int i = 0; i < 300; i++) {
      if (llama_decode(ctx, b) != 0) break;
      t = llama_sampler_sample(s, ctx, -1);
      if (llama_vocab_is_eog(vocab, t)) break;
      char buf[256];
      int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
      if (l > 0) out.append(buf, l);
      b = llama_batch_get_one(&t, 1);
    }
    llama_sampler_free(s);
  }
  jbyteArray r = env->NewByteArray(out.size());
  env->SetByteArrayRegion(r, 0, out.size(), (const jbyte*)out.data());
  return r;
}
