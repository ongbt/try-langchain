# https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa


from langchain_community.llms import LlamaCpp

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/ongbt/Downloads/llama-2-7b.Q5_K_M.gguf   ",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)

llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")