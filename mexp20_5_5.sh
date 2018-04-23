python scripts/train/few_shot/run_train.py --data.query 15 --data.dataset=miniImagenet --model.x_dim=3,84,84 --train.learning_rate 0.001 --data.way 20 --data.shot 5 --data.test_shot 5 --data.test_way 5 --data.cuda --log.exp_dir=results/m20_5way5shot 
python scripts/predict/few_shot/run_eval.py --data.test_shot 5 --data.test_way 5 --model.model_path=results/m20_5way5shot/best_model.t7
