from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, serve

def run_qwen3_4b_instruct_serve(
                          model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                          ):
    
    backend_config = TurbomindEngineConfig(dtype="float16",tp=1)


    pipe = serve(model_name, backend_config=backend_config)  # ho?c device="cpu" n?u không có GPU
    try:
        while True:
            pass  
    except KeyboardInterrupt:
        print("stop server.")

if __name__ == "__main__":
    run_qwen3_4b_instruct_serve(model_name = "Qwen/Qwen3-4B-Instruct-2507")
    
    

    
   

