import kagglehub
import pathlib

'''Example: 
    如果你的文件路径如下：
    final_project/
    ├──────────── file1.csv
    ├──────────── file2.csv
    ├──────────── check_path_exist.py
    └──────────── data/
                    ├──────────── file3.csv
                    └──────────── file4.csv

    那么你在check_path_exist.py中使用exits()方法：
    python
    >>>import pathlib
    >>>data_dir = pathlib.Path('./data')
    >>>data_dir.exists()
    True

    如果你在check_path_exist.py中使用iterdir()方法：
    python
    >>>import pathlib
    >>>data_dir = pathlib.Path('./data')
    >>>list(data_dir.iterdir())
    [PosixPath('final_project/data/file3.csv'), PosixPath('final_project/data/file4.csv')] 
    '''

def download_dataset():
    # Download latest version
    data_dir = pathlib.Path('./final_project/data')
    if data_dir.exists() and any(data_dir.iterdir()):
        print("Dataset already exists in the output directory. Skipping download.")
        return
    #方法详细解释： exists()方法用来判断路径是否存在，iterdir()方法用来列出当前路径下的所有内容
    else:
        print('Downloading dataset...')
        try:
            path = kagglehub.dataset_download("rounakbanik/the-movies-dataset", output_dir=data_dir)
            print("Path to dataset files:", path)
        except Exception as e:
            print(f"An error occurred while downloading the dataset: {e}")
            return
        
   


if __name__ == "__main__":
    download_dataset()