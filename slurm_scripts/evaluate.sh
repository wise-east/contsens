cd $PROJ_DIR/csci699_dcnlp_projectcode/

# constants
ONE_EPOCH_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt

TEN_EPOCH_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt 


ONE_EPOCH_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt

TEN_EPOCH_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt


for perturbation_strategy in sentence_shuffle sentence_drop sentence_reverse sentence_last random_text ; do 

    echo $perturbation_strategy 

    python contsens/evaluate_pretrained.py --perturbation_strategy $perturbation_strategy  --batch_size 32 --min_context 2 

done

# google/t5-v1_1-small - sentence_shuffle
# loss results: orig - 7.615 | perturbed - 7.602499 | pct diff: -0.170%

# google/t5-v1_1-small - sentence_drop
# loss results: orig - 7.615 | perturbed - 8.680026 | pct diff: 13.980%

# google/t5-v1_1-small - sentence_reverse
# loss results: orig - 7.615 | perturbed - 7.589117 | pct diff: -0.345%

# google/t5-v1_1-small - sentence_last | loss results: orig - 6.451 | perturbed - 9.424022 | pct diff: 46.097%

# google/t5-v1_1-small - random_text | loss results: orig - 7.615 | perturbed - 8.326671 | pct diff: 9.340%



for perturbation_strategy in sentence_shuffle sentence_drop sentence_reverse sentence_last random_text ; do 

    echo $perturbation_strategy 

    for model in $ONE_EPOCH_MODEL $TEN_EPOCH_MODEL $ONE_EPOCH_CONT4_MODEL $TEN_EPOCH_CONT4_MODEL ; do

        continue 
        echo $model 

        python contsens/evaluate_pretrained.py --perturbation_strategy $perturbation_strategy --from_ckpt $model --batch_size 32 --min_context 2

    done 
done 


# one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt - sentence_shuffle | loss results: orig - 2.356 | perturbed - 2.414494 | pct diff: 2.500%

# one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt - sentence_shuffle | loss results: orig - 2.557 | perturbed - 2.560026 | pct diff: 0.124%

# ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt - sentence_shuffle | loss results: orig - 2.110 | perturbed - 2.261489 | pct diff: 7.189%

# ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt - sentence_shuffle | loss results: orig - 2.318 | perturbed - 2.409678 | pct diff: 3.969%

# one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt - sentence_drop | loss results: orig - 2.356 | perturbed - 2.459769 | pct diff: 4.423%

# ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt - sentence_drop | loss results: orig - 2.110 | perturbed - 2.276945 | pct diff: 7.921%

# one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt - sentence_drop | loss results: orig - 2.557 | perturbed - 2.584779 | pct diff: 1.092%

# ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt - sentence_drop | loss results: orig - 2.318 | perturbed - 2.441622 | pct diff: 5.347%

# one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt - sentence_reverse | loss results: orig - 2.356 | perturbed - 2.450476 | pct diff: 4.028%

# ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt - sentence_reverse | loss results: orig - 2.110 | perturbed - 2.315454 | pct diff: 9.747%

# one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt - sentence_reverse | loss results: orig - 2.557 | perturbed - 2.560895 | pct diff: 0.158%

# ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt - sentence_reverse | loss results: orig - 2.318 | perturbed - 2.452350 | pct diff: 5.810%

# one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt - sentence_last | loss results: orig - 2.356 | perturbed - 2.474936 | pct diff: 5.066%

# ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt - sentence_last | loss results: orig - 2.110 | perturbed - 2.305318 | pct diff: 9.266%

# one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt - sentence_last | loss results: orig - 2.557 | perturbed - 2.614400 | pct diff: 2.250%

# ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt - sentence_last | loss results: orig - 2.318 | perturbed - 2.577366 | pct diff: 11.204%

# one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt - random_text | loss results: orig - 2.356 | perturbed - 2.766201 | pct diff: 17.431%

# ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt - random_text | loss results: orig - 2.110 | perturbed - 2.835537 | pct diff: 34.397%

# one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt - random_text | loss results: orig - 2.557 | perturbed - 2.749216 | pct diff: 7.523%

# ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt - random_text | loss results: orig - 2.318 | perturbed - 3.012052 | pct diff: 29.959%

