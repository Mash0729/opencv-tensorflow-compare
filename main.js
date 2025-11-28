const imageInput = document.getElementById("imageInput");
const clearButton = document.getElementById("clearButton");
const dispInput = document.getElementById("dispInput");

const nativeProcButton = document.getElementById("nativeProcButton");
const opencvProcButton = document.getElementById("opencvProcButton");
const tensorflowProcButton = document.getElementById("tensorflowProcButton");

const nativeResultCell = document.getElementById("nativeResultCell");
const opencvResultCell = document.getElementById("opencvResultCell");
const tensorflowResultCell = document.getElementById("tensorflowResultCell");

const nativeTimeCell = document.getElementById("nativeTimeCell");
const opencvTimeCell = document.getElementById("opencvTimeCell");
const tensorflowTimeCell = document.getElementById("tensorflowTimeCell");

let targetImage = null;

// 初期状態設定
imageInput.value = "";
disableProcButtons(true);

// 画像読み込み
imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) {
    dispInput.innerHTML = "";
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    dispInput.innerHTML = "";
    const img = document.createElement("img");
    img.src = e.target.result;
    dispInput.appendChild(img);

    targetImage = img;
    disableProcButtons(false);
  };
  reader.readAsDataURL(file);
});

// 実行ボタン押下時
nativeProcButton.addEventListener("click", () => {
  showResult(toSepiaWithNative, nativeResultCell, nativeTimeCell);
});

opencvProcButton.addEventListener("click", () => {
  showResult(toSepiaWithOpenCV, opencvResultCell, opencvTimeCell);
});

tensorflowProcButton.addEventListener("click", () => {
  showResult(toSepiaWithTensorflow, tensorflowResultCell, tensorflowTimeCell);
});

// セピア化: NativeなJavaScriptによる処理
function toSepiaWithNative() {
  // 画質劣化防止のために元画像の解像度のcanvasを作成
  const fullCanvas = document.createElement("canvas");
  fullCanvas.width = targetImage.naturalWidth;
  fullCanvas.height = targetImage.naturalHeight;
  // オフスクリーンで描画
  const ctx = fullCanvas.getContext("2d");
  ctx.drawImage(targetImage, 0, 0, fullCanvas.width, fullCanvas.height);

  // 画像データの取得
  const imageData = ctx.getImageData(0, 0, fullCanvas.width, fullCanvas.height);
  const data = imageData.data;

  console.time("Native");
  // 1次元配列dataにはRGBAの順で0~255の整数が格納されている
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i],
      g = data[i + 1],
      b = data[i + 2];

    data[i] = r * 0.393 + g * 0.769 + b * 0.189;
    data[i + 1] = r * 0.349 + g * 0.686 + b * 0.168;
    data[i + 2] = r * 0.272 + g * 0.534 + b * 0.131;
  }
  console.timeEnd("Native");

  ctx.putImageData(imageData, 0, 0);
  return fullCanvas;
}

// セピア化: OpenCV.jsによる処理
function toSepiaWithOpenCV() {
  let src;
  let floatSrc;
  let mat;
  let floatDst;
  let dst;
  let outputCanvas = null;

  try {
    // 画質劣化防止のために元画像の解像度のcanvasを作成
    const fullCanvas = document.createElement("canvas");
    fullCanvas.width = targetImage.naturalWidth;
    fullCanvas.height = targetImage.naturalHeight;
    // オフスクリーンで描画
    const ctx = fullCanvas.getContext("2d");
    ctx.drawImage(targetImage, 0, 0, fullCanvas.width, fullCanvas.height);

    // Mat形式で画像を取得
    src = cv.imread(fullCanvas);

    // 32bit-floatへの一時的な変換
    // CV_8Uのままではクリッピングが起こる
    floatSrc = new cv.Mat();
    src.convertTo(floatSrc, cv.CV_32F);

    // セピア化用の行列
    // RGBA->RGBAへの変換
    // prettier-ignore
    const sepiaMatrix = [
      0.393, 0.769, 0.189, 0,
      0.349, 0.686, 0.168, 0,
      0.272, 0.534, 0.131, 0,
      0, 0, 0, 1,
    ];
    // 32bit-floatに変換
    mat = cv.matFromArray(4, 4, cv.CV_32F, sepiaMatrix);
    floatDst = new cv.Mat();
    console.time("OpenCV.js");
    cv.transform(floatSrc, floatDst, mat); // 線形変換
    console.timeEnd("OpenCV.js");

    // 表示用に8bit符号なし整数型に戻す
    dst = new cv.Mat();
    floatDst.convertTo(dst, cv.CV_8U);

    // 表示用のcanvasを作って描画
    outputCanvas = document.createElement("canvas");
    cv.imshow(outputCanvas, dst);
  } catch (err) {
    console.error(err);
  } finally {
    // cv.MatなどのインスタンスにはGCが効かないのでメモリ解放を自分で行う
    if (src) src.delete();
    if (floatSrc) floatSrc.delete();
    if (mat) mat.delete();
    if (floatDst) floatDst.delete();
    if (dst) dst.delete();
  }

  return outputCanvas;
}

// セピア化: Tensorflow.jsによる処理
async function toSepiaWithTensorflow() {
  const outputCanvas = document.createElement("canvas");

  // 画質劣化防止のために元画像の解像度のcanvasを作成
  const fullCanvas = document.createElement("canvas");
  fullCanvas.width = targetImage.naturalWidth;
  fullCanvas.height = targetImage.naturalHeight;
  // オフスクリーンで描画
  const ctx = fullCanvas.getContext("2d");
  ctx.drawImage(targetImage, 0, 0, fullCanvas.width, fullCanvas.height);

  // テンソルは、ベクトルや行列の高次元への一般化
  // ref: https://www.tensorflow.org/js/guide/tensors_operations?hl=ja

  // <Tensor>.dispose()でメモリ解放を行う代わりにtidyメソッドを使う

  const resultTensor = tf.tidy(() => {
    const imageTensor = tf.browser
      .fromPixels(fullCanvas)
      .slice([0, 0, 0], [-1, -1, 3]); // Aチャンネル除去

    // セピア化用の行列
    const sepiaMatrix = tf
      .tensor2d([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
      ])
      .transpose();

    // 32bit-float化
    const reshapedTensor = imageTensor.toFloat().reshape([-1, 3]);
    console.time("Tensorflow.js");
    const sepiaTensor = reshapedTensor.matMul(sepiaMatrix);
    console.timeEnd("Tensorflow.js");

    const newImageTensor = sepiaTensor
      .clipByValue(0, 255)
      .reshape(imageTensor.shape);
    return newImageTensor;
  });
  // toPixelsは非同期関数で、tf.tidy内での非同期関数実行は非推奨のため、外で処理を行う
  await tf.browser.toPixels(resultTensor.toInt(), outputCanvas);
  resultTensor.dispose();

  return outputCanvas;
}

// 実行ボタンの無効化
function disableProcButtons(disabled) {
  nativeProcButton.disabled = disabled;
  opencvProcButton.disabled = disabled;
  tensorflowProcButton.disabled = disabled;
}

// 特定の処理を行って結果をresultCellに描画、実行時間をtimeCellに記録する関数
async function showResult(func, resultCell, timeCell) {
  const startTime = performance.now();
  const resultCanvas = await func();
  const endTime = performance.now();

  if (resultCanvas) {
    resultCell.innerHTML = "";
    resultCell.appendChild(resultCanvas);

    const time = endTime - startTime;
    timeCell.textContent = time.toFixed(2);
  }
}

// 表内のすべてのセルをクリア
function clearAllCells() {
  const resultCells = document.querySelectorAll(".resultCell");
  const timeCells = document.querySelectorAll(".timeCell");

  const toEmpty = (cell) => {
    cell.innerHTML = "";
  };

  resultCells.forEach(toEmpty);
  timeCells.forEach(toEmpty);
}
