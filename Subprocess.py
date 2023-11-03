import subprocess

def extract_cmd_output(cmd, output_file):
    with open(output_file, "w") as file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        for line in process.stdout:
            line = line.decode().strip()
            print(line)
            file.write(line + "\n")

# Đường dẫn và tên tệp tin đầu ra
output_directory = "D:/dqd/KAIR-1.1/Result"
output_file = output_directory + "result_main_train.txt"

# Lệnh để chạy các tệp main_train_
# Thay đổi lệnh tương ứng với tệp main_train_ mà bạn muốn chạy
cmd = "python main_train_psnr.py --opt options/train_msrresnet_psnr.json"

# Trích xuất kết quả và ghi vào tệp tin
extract_cmd_output(cmd, output_file)
