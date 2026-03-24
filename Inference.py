import os
import argparse
from types import SimpleNamespace
from beta_vae import * 
import pandas as pd
import yaml
from General_utils import read_one_column_tsv

def create_parser():
    parser = argparse.ArgumentParser(
        description="Run inference using a trained VAE model"
    )

    parser.add_argument(
        "-m", "--model_dir",
        type=str,
        required=True,
        help="Path to folder where the VAE model is stored"
    )

    parser.add_argument(
        "-r", "--region",
        type=str,
        required=True,
        help="Region"
    )

    parser.add_argument(
        "-c", "--criteria",
        type=str,
        required=True,
        help="Segmentation criteria (e.g., Comm, DWM, SWM)"
    )

    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to data (crops)"
    )

    parser.add_argument(
        "-o", "--outdir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "-s", "--subjects",
        type=str,
        required=True,
        help=".tsv file listing the subjects to process"
    )

    return parser


def run(model_dir, region, criteria, outdir, subjects, data):
    print("=== Running VAE Inference ===")
    print(f"Model dir: {model_dir}")
    print(f"Region: {region}")
    print(f"Criteria: {criteria}")
    print(f"Output dir: {outdir}")

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)


    torch.manual_seed(0)
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)

    # ---- Example: locate model checkpoint ----
    # (adapt depending on your naming)
    checkpoint_path = os.path.join(model_dir, "model.pt")
    config_path     = os.path.join(model_dir, "config.yaml")

    # ---- Sanity check ---- 
    if not os.path.exists(checkpoint_path):
        print(f"Model not found: {checkpoint_path}")
        return
    else:
        print(f"Found model: {checkpoint_path}")

    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return
    else:
        print(f"Config found: {config_path}")

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)
    print("--- Inference ---")

    # ---- loading model ----
    model = VAE(config.in_shape, config.n, depth=config.depth, loss_selected= config.loss)
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model = model.to(device)
    model.eval()

    if config.loss == 'CrossEntropy':
        weights = [1, 2]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
        print('"""Using Cross Entropy Loss, reduction=sum')

    elif config.loss == 'MSE':
        criterion = nn.MSELoss( reduction='sum')
        print('"""Using MSE Loss, reduction=sum')

    config.path_crops = data

    subjects_list = read_one_column_tsv(subejcts)
    subjects_set  = create_subset_from_list(config,subjects_list)
    start_loading = time.time()
    subjects_loader = torch.utils.data.DataLoader(subjects_set, batch_size=1,num_workers=6, shuffle=False)
    print(f"{bcolors.MAGENTA}-- -Created subjects loader in  {time.time() - start_loading} seconds ---{bcolors.RESET}")

    embeddings_list = []
    recon_error_list = []
    subs_ids = []

    with torch.no_grad():

        for inputs, path in subjects_loader: #We iterate the subjects dataset one by one
            print(path)
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(inputs, dim=1).long()
            z, logvar = model.encode(inputs) # z = mean because no random sampling
            outputs = model.decode(z)
           

            if config.loss == 'CrossEntropy':
                recon_loss_val, kl_val, loss_val = vae_loss(outputs, target, z, logvar, config.loss_func,
                                kl_weight=config.kl_weight) 
                outputs = torch.argmax(outputs, dim=1) 

            elif config.loss == 'MSE':
                recon_loss_val, kl_val, loss_val = vae_loss(outputs, inputs, z, logvar, config.loss_func,
                                kl_weight=config.kl_weight) 
            embeddings.append(z.cpu().detach().numpy())
            recon_error_lists.append(recon_loss_val.cpu().detach().numpy())
            subs_ids.append(path)

    
    embeddings = np.asarray(embeddings_list)
    recon_error = np.asarray(recon_error_list)

    n = embeddings.shape[1]

    # ---- Embeddings dataframe ----
    columns = ["subject"] + [f"dim_{i}" for i in range(n)]

    df_embeddings = pd.DataFrame(
        data=[[subj] + emb.tolist() for subj, emb in zip(subjects, embeddings)],
        columns=columns
    )

    df_embeddings.to_csv(f"{outdir}/embeddings.csv", index=False)


    # ---- Reconstruction error dataframe ----
    # Case 1: scalar error per subject
    df_recon = pd.DataFrame({
        "subject": subjects,
        "recon_error": recon_error})

    df_recon.to_csv(f"{outdir}/recon_error.csv", index=False)

def main():
    parser = create_parser()
    args = parser.parse_args()

    run(
        model_dir=args.model_dir,
        region=args.region,
        criteria=args.criteria,
        outdir=args.outdir,
        subjects=args.subjects,
        data=args.data
    )


if __name__ == "__main__":
    main()