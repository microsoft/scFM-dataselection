import datetime
import sys
import os
import pickle
from pathlib import Path
import subprocess
import logging
sys.path.append('./Geneformer/')
from geneformer import Classifier
from geneformer import classifier_utils as cu
from geneformer import perturber_utils as pu
from datasets import Dataset, load_from_disk

def fine_tune_geneformer_celltype_classification(geneformer_model_type, output_dir, cell_type_column, input_data_file_train, input_data_file_test, pretrained_model, input_data_file_val):

    #current_date = datetime.datetime.now()
    #datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
    geneformer_model_type = geneformer_model_type

    output_prefix = "cm_classifier"
    output_dir = output_dir + f"/{geneformer_model_type}" 
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)


    training_args = {
        "num_train_epochs": 1.0,
        "learning_rate": 0.000210031,
        "lr_scheduler_type": "linear",
        "warmup_steps": int(1000),
        "weight_decay": 0.233414,
        "per_device_train_batch_size": 18,
        "seed": 25,
    }

    # Print the type of each argument in the training_args dictionary
    for key, value in training_args.items():
        print(f"The type of '{key}' is {type(value).__name__}")

    cc = Classifier(classifier="cell",
                    cell_state_dict={"state_key": str(cell_type_column), "states": "all"},
                    filter_data=None,
                    training_args=training_args,
                    max_ncells=None,
                    freeze_layers=2,
                    num_crossval_splits=1,
                    forward_batch_size=200,
                    nproc=16,
                    no_eval=True,
                    ngpu=8)

    print('Preparing train data...') #####################################################################################################

    print('0', output_dir)

    data = pu.load_and_filter(cc.filter_data, cc.nproc, input_data_file=input_data_file_train)
    if "label" in data.features:
        print(
            "Column name 'label' must be reserved for class IDs. Please rename column."
                )
        
    # rename cell state column to "label"
    data = cu.rename_cols(data, cc.cell_state_dict["state_key"])

    # convert classes to numerical labels and save as id_class_dict
    data, id_class_dict = cu.label_classes(
        cc.classifier, data, cc.gene_class_dict, cc.nproc
        )
    # save id_class_dict for future reference
    id_class_output_path = output_dir+'/'+ f"{output_prefix}_id_class_dict_train.pkl"
    
    with open(id_class_output_path, "wb") as f:
        pickle.dump(id_class_dict, f)


    data_output_path = output_dir +'/'+ f"{output_prefix}_labeled_train.dataset"

    data.save_to_disk(str(data_output_path))


    print('Preparing test data...') #####################################################################################################
    data = pu.load_and_filter(cc.filter_data, cc.nproc, input_data_file=input_data_file_test)
    if "label" in data.features:
        print(
            "Column name 'label' must be reserved for class IDs. Please rename column."
                )
    # rename cell state column to "label"
    data = cu.rename_cols(data, cc.cell_state_dict["state_key"])
        # convert classes to numerical labels and save as id_class_dict
    data, id_class_dict = cu.label_classes(
        cc.classifier, data, cc.gene_class_dict, cc.nproc
        )
    # save id_class_dict for future reference
    id_class_output_path = output_dir+'/'+ f"{output_prefix}_id_class_dict_test.pkl"
    with open(id_class_output_path, "wb") as f:
            pickle.dump(id_class_dict, f)

    data_output_path = output_dir +'/'+ f"{output_prefix}_labeled_test.dataset"
    data.save_to_disk(str(data_output_path))

    print('Preparing val data...') #####################################################################################################
    data = pu.load_and_filter(cc.filter_data, cc.nproc, input_data_file=input_data_file_val)
    if "label" in data.features:
        print(
            "Column name 'label' must be reserved for class IDs. Please rename column."
                )
    # rename cell state column to "label"
    data = cu.rename_cols(data, cc.cell_state_dict["state_key"])
        # convert classes to numerical labels and save as id_class_dict
    data, id_class_dict = cu.label_classes(
        cc.classifier, data, cc.gene_class_dict, cc.nproc
        )
    # save id_class_dict for future reference
    id_class_output_path = output_dir +'/'+ f"{output_prefix}_id_class_dict_val.pkl"
    with open(id_class_output_path, "wb") as f:
            pickle.dump(id_class_dict, f)

    data_output_path = output_dir +'/'+ f"{output_prefix}_labeled_val.dataset"
    data.save_to_disk(str(data_output_path))



    print("Finetuning model...") #####################################################################################################

    file_path = f"{output_dir}/{output_prefix}_id_class_dict_train.pkl"
    with open(file_path, 'rb') as file:
    # Load the dictionary from the file
        id_class_dict = pickle.load(file)

    num_classes = cu.get_num_classes(id_class_dict=id_class_dict)

    train_data=f"{output_dir}/{output_prefix}_labeled_train.dataset"
    eval_data=f"{output_dir}/{output_prefix}_labeled_val.dataset"
    train_data = load_from_disk(train_data)
    eval_data = load_from_disk(eval_data)

    # trainer = cc.hyperopt_classifier(
    #     model_directory=os.path.abspath(pretrained_model),
    #     num_classes=num_classes,
    #     train_data=train_data,
    #     eval_data=eval_data,
    #     output_directory=output_dir,
    #     n_trials=n_trials)

    training_args = {
        "num_train_epochs": 1.0,
        "learning_rate": 0.000210031,
        "lr_scheduler_type": "polynomial",
        "warmup_steps": 1000,
        "weight_decay": 0.233414,
        "per_device_train_batch_size": 18,
        "seed": 25,
    }

    print(training_args)
    # Print the type of each argument in the training_args dictionary
    for key, value in training_args.items():
        print(f"The type of '{key}' after prep data is {type(value).__name__}")
    
    # cc.train_classifier(
    #     model_directory=os.path.abspath(pretrained_model),
    #     num_classes=num_classes,
    #     train_data=train_data,
    #     eval_data=eval_data,
    #     output_directory=output_dir,
    #     predict=True,
    #)

 
    #model_directory=os.path.abspath(pretrained_model),
    # prepared_input_data_file=train_data,
    # eval_data_file=eval_data,  # New parameter for eval dataset
    # id_class_dict_file=id_class_dict,
    output_directory=output_dir,
    # output_prefix=output_prefix,
    predict_eval=True,
    #n_hyperopt_trials=0,

    # num_crossval_splits = 0

    """
    Validate the model using a specific evaluation dataset provided by the user.
    """
    # logger = logging.getLogger(__name__)

    # if num_crossval_splits != 1:
    #     logger.error("This method currently supports only a single evaluation dataset.")
    #     raise ValueError("num_crossval_splits must be 1")

    # Load ID to class dictionary
    id_class_dict = id_class_dict
    # class_id_dict = {v: k for k, v in id_class_dict.items()}

    # Load the training data
    train_data = train_data

    # Load the evaluation data
    eval_data = eval_data

    # Setup output directory
    # current_date = datetime.datetime.now()
    # datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    # if output_directory[-1:] != "/":
    #     output_directory += "/"
    output_dir = output_dir
    subprocess.call(f"mkdir -p {output_dir}", shell=True)

    # Number of classes
    num_classes = num_classes

    # Training or Hyperparameter Optimization
    #if n_hyperopt_trials == 0:
    trainer = cc.train_classifier(
        model_directory=os.path.abspath(pretrained_model), num_classes=num_classes, train_data=train_data, eval_data=eval_data, output_directory=output_dir, predict=False
        )
    # else:
    #     trainer = cc.hyperopt_classifier(
    #         model_directory=os.path.abspath(pretrained_model), num_classes=num_classes, train_data=train_data, eval_data=eval_data, output_directory=output_dir, n_trials=n_hyperopt_trials
    #     )

    # Evaluate the model
    result = cc.evaluate_model(
        trainer.model, num_classes, id_class_dict, eval_data, predict_eval, output_dir, output_prefix
    )

    # Optionally save results or do additional processing...
    print("Validation complete with results:", result)

    return result


def main():
    print(sys.argv)
    geneformer_model_type = sys.argv[1]
    output_dir = sys.argv[2]
    cell_type_column = sys.argv[3]
    input_data_file_train = sys.argv[4]  # Path to directory containing .dataset input
    input_data_file_test = sys.argv[5]  # Path to directory containing .dataset input
    pretrained_model = sys.argv[6] #Path to Geneformer model
    input_data_file_val = sys.argv[7] #Path to Geneformer mode1
    # n_trials = int(sys.argv[7]) 

    # pretrained_dir = Path(pretrained_model)
    # #pretrained_model = str(list(pretrained_dir.glob("models/*/models"))[0])
    # pretrained_model = str(list(pretrained_dir.glob("models/*/models"))[-1])
    pretrained_model = pretrained_model

    #print('fine tuning pretrained model:' , pretrained_model)

    fine_tune_geneformer_celltype_classification(geneformer_model_type, output_dir, cell_type_column, input_data_file_train, input_data_file_test, pretrained_model, input_data_file_val)

if __name__ == "__main__":
    main()