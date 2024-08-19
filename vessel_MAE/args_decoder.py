import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    parser.add_argument("--lr", default=0.0001,type=float)##??
    parser.add_argument("--batch_size", default=16, type=int)##??
    parser.add_argument("--num_epochs", default=2000,type=int)##??
    parser.add_argument("--patch_size", default=(16,32), type=tuple)##??
    parser.add_argument("--mask_ratio", default=0.5,type=float)##??
    parser.add_argument("--loss_ratio", default=0.5,type=float)##??
    parser.add_argument("--data_path", default="/root/work/data/2021", type=str)##??
    parser.add_argument("--result_path", default="/root/work/result/decoder_feature", type=str)##??
    parser.add_argument("--random_seed", default=0, type=int)##??
    parser.add_argument("--random_state", default=1, type=int)##??


    args = parser.parse_args()

    return args

def get_name():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## use for visualise ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--model_name", default="ilma", help="[ilma,irra]")
    parser.add_argument("--config_file", default="./xxx/config_file", help="./xxx/config_file")
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()
    return args