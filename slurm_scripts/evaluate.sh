cd $PROJ_DIR/csci699_dcnlp_projectcode/

# constants
ONE_EPOCH_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch/t5-v1_1-small_epoch=0-step=9507-val_loss=2.48.ckpt

TEN_EPOCH_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch/t5-v1_1-small_epoch=6-step=66549-val_loss=2.19.ckpt 


ONE_EPOCH_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch_mincontext=4/t5-v1_1-small_epoch=0-step=4473-val_loss=2.68.ckpt

TEN_EPOCH_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=4/t5-v1_1-small_epoch=7-step=35784-val_loss=2.25.ckpt

TEN_EPOCH_LMADAPTED_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch/t5-small-lm-adapt_epoch=9-step=95070-val_loss=2.09.ckpt

TEN_EPOCH_LMADAPTED_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=4/t5-small-lm-adapt_epoch=9-step=44730-val_loss=2.12.ckpt

TEN_EPOCH_LMADAPTED_CONT6_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=6/t5-small-lm-adapt_epoch=9-step=27890-val_loss=2.11.ckpt

TEN_EPOCH_LMADAPTED_CONT8_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=8/t5-small-lm-adapt_epoch=8-step=14292-val_loss=2.14.ckpt


TEN_EPOCH_LMADAPTED_CONT10_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=10/t5-small-lm-adapt_epoch=9-step=8200-val_loss=2.17.ckpt

TEN_EPOCH_LMADAPTED_CONT12_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/ten_epoch_mincontext=12/t5-small-lm-adapt_epoch=7-step=3240-val_loss=2.24.ckpt




ONE_EPOCH_LMADAPTED_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch/t5-small-lm-adapt_epoch=0-step=9507-val_loss=2.28.ckpt

ONE_EPOCH_LMADAPTED_CONT4_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch_mincontext=4/t5-small-lm-adapt_epoch=0-step=4473-val_loss=2.29.ckpt

ONE_EPOCH_LMADAPTED_CONT8_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch_mincontext=8/t5-small-lm-adapt_epoch=0-step=1588-val_loss=2.28.ckpt

ONE_EPOCH_LMADAPTED_CONT12_MODEL=$PROJ_DIR/csci699_dcnlp_projectcode/results/one_epoch_mincontext=12/t5-small-lm-adapt_epoch=0-step=405-val_loss=2.42.ckpt


SCRATCH=$PROJ_DIR/csci699_dcnlp_projectcode/results/scratch/t5-small-lm-adapt_epoch=9-step=95070-val_loss=3.01.ckpt

for perturbation_strategy in "all" ; do 
    continue 
    echo $perturbation_strategy 

    for dataset in daily_dialog c4 ; do 

        python contsens/evaluate_pretrained.py --perturbation_strategy $perturbation_strategy  --batch_size 32 --min_context 5 --dataset $dataset --use_spacy

        # python contsens/evaluate_pretrained.py --perturbation_strategy $perturbation_strategy  --batch_size 32 --min_context 5 --model "google/t5-small-lm-adapt" --dataset $dataset --use_spacy

    done 
done


for perturbation_strategy in "all" ; do 
    # continue 
    echo $perturbation_strategy 

    # for model in $ONE_EPOCH_MODEL $TEN_EPOCH_MODEL $ONE_EPOCH_CONT4_MODEL $TEN_EPOCH_CONT4_MODEL ; do
    # for model in $TEN_EPOCH_MODEL ; do 

    # for model in $TEN_EPOCH_LMADAPTED_MODEL ; do 
    # for model in $SCRATCH ; do 
    for model in $TEN_EPOCH_LMADAPTED_CONT12_MODEL ; do 

        echo $model 

        # for dataset in daily_dialog c4 ; do 
        for dataset in daily_dialog ; do 

            python contsens/evaluate_pretrained.py --perturbation_strategy $perturbation_strategy --from_ckpt $model --batch_size 32 --min_context 5 --use_spacy --dataset $dataset 
        done 
    done 
done 

