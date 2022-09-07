FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y \
    gcc git wget \
    ffmpeg libsm6 libxext6 \
    python3.8 python3.8-dev python3-pip
    
RUN git clone https://github.com/nadeemlab/DoseRTX.git
WORKDIR DoseRTX
#RUN pip install -r requirements.txt
RUN pip install dominate==2.7.0 GPUtil==1.4.0 matplotlib==3.5.3 numpy==1.23.2 pandas==1.4.4 Pillow==9.2.0 pydicom==2.3.0 SimpleITK==2.2.0 scikit-image==0.17.2 torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 visdom==0.1.8.9
RUN git submodule init && git submodule update
RUN wget -q https://zenodo.org/record/7036132/files/sample_nrrd_data.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/sample_patient_data.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/sample_processed_data.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/sample_predictions_for_loss_functions.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/MAE_DVH_loss.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/MAE_loss.tar.bz2 \
    && wget -q https://zenodo.org/record/7036132/files/MAE_Moment_loss.tar.bz2 \
    && tar xjf sample_patient_data.tar.bz2 \
    && tar xjf sample_nrrd_data.tar.bz2 \
    && tar xjf sample_processed_data.tar.bz2 \
    && tar xjf MAE_DVH_loss.tar.bz2 \
    && tar xjf MAE_loss.tar.bz2 \
    && tar xjf MAE_Moment_loss.tar.bz2 \
    && tar xjf sample_predictions_for_loss_functions.tar.bz2 && rm *.bz2
RUN mkdir -p  sample_processed_data/test && mv sample_processed_data/*.npz sample_processed_data/test
RUN mkdir -p checkpoints && mv MAE_DVH_loss/ checkpoints/ \
    && mv MAE_Moment_loss/ checkpoints/ \
    && mv MAE_loss/ checkpoints/
