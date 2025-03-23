# poetry run gpt experiment configs/configs_model.json
# poetry run gpt evaluate results_exp1 scores_exp1

poetry run gpt experiment configs/configs_hyperparams.json
poetry run gpt evaluate results_exp2 scores_exp2

poetry run gpt experiment configs/configs_final.json
poetry run gpt evaluate results_exp3 scores_exp3