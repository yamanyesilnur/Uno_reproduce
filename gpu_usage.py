from datetime import datetime
from pynvml import *
import time


def print_nvidia_gpu_status_on_log_file(log_file, delay_in_seconds):
    with open(log_file, "w") as log:
        nvmlInit()
        log.write(f"Driver Version: {nvmlSystemGetDriverVersion()}\n")
        deviceCount = nvmlDeviceGetCount()
        log.write(f"I found out {deviceCount} GPUs:\n")
        gpus = []
        for i in range(deviceCount):
            gpus.append(nvmlDeviceGetHandleByIndex(i))
            log.write(f"\tDevice {i} : {nvmlDeviceGetName(gpus[-1])}\n")
        used_total = {i: 0 for i in range(len(gpus))}
        total_total = {i: 0 for i in range(len(gpus))}
        free_total = {i: 0 for i in range(len(gpus))}
        while True:
            for i in range(100):
                for i, gpu in enumerate(gpus):
                    info = nvmlDeviceGetMemoryInfo(gpu)
                    used_total[i] += info.used / 10**9
                    total_total[i] += info.total / 10**9
                    free_total[i] += info.free / 10**9
                time.sleep(delay_in_seconds / 100)
            for i, _ in enumerate(gpus):
                used_total[i] /= 100
                total_total[i] /= 100
                free_total[i] /= 100
            now = datetime.now()
            current_time = now.strftime("%d_%m_%Y_%H:%M:%S")
            log.write(f"{current_time}:\n")
            for i, _ in enumerate(gpus):
                log.write(
                    f"\t{used_total[i]:.2f} GB / {total_total[i]:.2f} GB [free: {free_total[i]:.2f} GB]\n"
                )
                log.flush()
