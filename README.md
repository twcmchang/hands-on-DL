# 手把手的深度學習實務
## 課前準備: 安裝 Keras
0. Python 建議安裝 python3.5+ 的版本。請依照各自的環境，參照下方安裝說明，安裝好所需要的 packages。
1. [For Linux or MacOSX] 安裝 Keras，請依 https://github.com/tw-cmchang/hand-on-dl/blob/master/keras_installation.pdf 安裝說明。
2. [For Windows] 請參考 https://github.com/chihfanhsu/hand-on-dl 安裝說明。
2. 下載 https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py 這份範例程式並執行測試。
```python
python mnist_mlp.py
```
+ 若能成功執行，表示安裝成功。
+ 所需套件整理如下:
```
pip install numpy scipy future matplotlib scikit-learn
pip install keras
```
+ 由於作業系統環境因人而異，若有任何安裝上的問題，請不吝來信詢問: cmchang@iis.sinica.edu.tw，謝謝大家的海涵，希望各位能在 12/16 上課前完成安裝流程，謝謝！

## 下載程式、投影片
1. 請於 12/16 上課前至 https://goo.gl/UmVMMZ 下載所需的程式碼與投影片。
2. 課前也會將最新版本更新到 [slideshare](http://www.slideshare.net/tw_dsconf/ss-70083878)。

## (optional) 安裝運算加速庫
1. 下載並執行 https://github.com/tw-cmchang/hand-on-dl/blob/master/checkblas.py 測試是否有安裝運算加速庫。
2. 安裝 Anaconda 的學員，也確認是否有通過 checkblas.py 測試。儘管 Anaconda 內有 mkl 運算加速庫。
3. 安裝 openblas，請依 https://github.com/tw-cmchang/hand-on-dl/blob/master/openblas_installation.pdf 。
4. 另外，若需要安裝 openblas 可至 https://drive.google.com/drive/folders/0ByfnsehogjWtbndTY3JncE95bjQ 下載 (謝謝 Chih-Fan)。

## (optional) GPU 安裝 (需要 NVIDIA 顯示卡)
### 在 Windows 10 安裝 CUDA & cuDNN 可以參考下列網址
1. [安裝 CUDA&Theano](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/)
2. [安裝 cuDNN](http://ankivil.com/making-theano-faster-with-cudnn-and-cnmem-on-windows-10/)

### 在 ubuntu 上安裝可以參考下列影片，建議安裝 CUDA 7.5
* https://www.youtube.com/watch?v=wjByPfSFkBo

### 沒有 GPU 的折衷方案 (Windows 10, openBLAS CPU 加速)
* 請安裝 [openBLAS](https://github.com/chihfanhsu/dnn_hand_by_hand/blob/master/openblas_install.pdf)

## Other Questions
+ 有學員回報在 win10 安裝 Anaconda2 後，使用 pip install theano/ pip install keras 出現下方錯誤訊息：
```pyhon
UnicodeDecodeError: 'ascii' codec can't decode byte 0xb8 in position 0: ordinal not in range(128)
```
此為 Anaconda2 的預設編碼問題。請在 Anaconda2\Lib\site-packages 裡增加一個 sitecustomize.py，內容如下：
```python
import sys 
sys.setdefaultencoding('gbk')
```
之後在 pip install keras 試試看，若有問題請再來信。謝謝該位熱心的同學提供解法 :)。
