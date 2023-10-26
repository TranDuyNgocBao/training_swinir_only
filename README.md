# Clone repo
```
Bước 1: git clone https://github.com/LamKser/esrgan_main.git
Bước 2: Di chuyển tới thư mục esrgan_main
        cd esrgan_main
```

# Requirements
Cài đặt thư viện
```
pip install -r requirements.txt
```

Bên trong thư mục `esrgan_main`:
# Data
* Tải `dataset.zip` tại link: [GG drive](https://drive.google.com/file/d/1slUHOB8UzCznLFMwfwmzO6_oktbiuvK-/view?usp=sharing)
* Tiếp theo giải nén data vào thư mục `dataset`
* Cấu trúc của thư mục `dataset` sẽ bao gồm:
    ```
    dataset
    ├───combine_data_training   // data training
    └───val                     // data val (5 ảnh từ tập test USR248)
        ├───hr
        └───lr
    ```

# Pretrain model
* Tải `pretrain.zip` tại link: [GG drive](https://drive.google.com/file/d/1T2uQxyzWZPfnMdSYVrD8Jc9w8Dj79PG7/view?usp=sharing)
* Tiếp theo giải nén data vào thư mục `esrgan/model_zoo`
* Cấu trúc của thư mục `esrgan/model_zoo` sẽ bao gồm:
    ```
    esrgan/model_zoo
            ├───rrdb_x4_esrgan.pth  
            └───rrdb_x4_psnr.pth
    ```
# Training

* Train original ESRGAN
```
python esrgan/main_train_gan.py --opt esrgan/options/train_rrdb_esrgan.json
```
* Train ESRGAN with patch Gan
```
python esrgan/main_train_psnr.py --opt esrgan/options/train_rrdb_esrgan_patchgan.json
```

* Train PSNR
```
python esrgan/main_train_psnr.py --opt esrgan/options/train_rrdb_psnr.json
```

**Note:** Các trọng số được lưu tại thư mục `experiments`
