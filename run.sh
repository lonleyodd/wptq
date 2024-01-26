model_name=$1

case $model_name in
    "vit")
    bash vit/quan_eval.sh
    ;;
    "opt")
    python  opt.py
    ;;
    "llama2")
    deepspeed --num_gpus 2 llama2.py 
    ;;
    *)
    echo "unsupported model:" $model_name 
    ;;
esac