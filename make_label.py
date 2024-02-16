import os
import argparse

def create_label_files(base_path):
    # base_path에서 모든 하위 폴더 탐색
    folders = [f.name for f in os.scandir(base_path) if f.is_dir()]

    # labels 폴더 생성
    labels_dir = os.path.join(base_path, 'labels')
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # 각 폴더에 대해 작업 수행
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        
        # labels 폴더에 동일한 이름의 txt 파일 생성
        for img in images:
            txt_name = os.path.splitext(img)[0] + '.txt'
            txt_path = os.path.join(labels_dir, txt_name)
            with open(txt_path, 'w') as txt_file:
                txt_file.write(folder)

    print('Label files have been created successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate label txt files for images based on their folder names.')
    parser.add_argument('--path', type=str, required=True, help='Base path containing the image folders and where the labels folder will be created.')
    args = parser.parse_args()

    create_label_files(args.path)