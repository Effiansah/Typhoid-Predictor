import uuid
import subprocess
import pandas as pd


def generate_molecular_descriptors(smiles='CCCCC1=CC(=O)OC2=C1C=CC(=C2CN3CCCC3)O'):
    """
    Generate molecular descriptors for a given SMILES string.
    :param smiles: SMILES string
    :return: filename containing molecular descriptors
    """
    error = False
    filename = str(uuid.uuid4())[:6]

    with open(filename+'.smiles', 'w') as f:
        f.write(smiles)
    try:
        openbabel_command = f"obabel {filename}.smiles -O {filename}.sdf --gen2d"
        subprocess.run(openbabel_command, shell=True)
        #mold2_command = f"/Users/user/python/salutem/fyp/mold2/Mold2 -i {filename}.sdf -o {filename}.txt"
        mold2_command = "./Mold2 -i {filename}.sdf -o {filename}.txt"

        # intentionally setting the process to timeout after one second because the mold2 executable
        # requires user input to continue. This is a hacky way to get around that.
        # we catch the timeout exception and delete the sdf and smiles files, leaving only the txt file containing the descriptors
        x = subprocess.run(mold2_command, shell=True, timeout=2)

    except subprocess.TimeoutExpired:
        print("done. deleting smiles and sdf files...")
        subprocess.run(f"rm ./{filename}.smiles", shell=True)
        subprocess.run(f"rm ./{filename}.sdf", shell=True)
        subprocess.run(f"rm ./report.txt", shell=True)
        print("deleted smiles, sdf and report files...")
    except:
        error = True
        print("error occured")
        subprocess.run(f"rm ./{filename}.smiles", shell=True)
        subprocess.run(f"rm ./{filename}.sdf", shell=True)
        subprocess.run(f"rm ./report.txt", shell=True)
        print("deleted smiles, sdf and report files...")

    return f"{filename}.txt"
