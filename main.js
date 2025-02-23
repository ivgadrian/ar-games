// import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";


const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const canvas2 = document.getElementById("canvas2");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const ctx2 = canvas2.getContext("2d");
let session = null;
launch();

async function launch() {
    session = await loadModel();
    await startCamera();
}


async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas2.width = video.videoWidth;
        canvas2.height = video.videoHeight;

        drawFrame()
    };
}

function drawFrame() {
    ctx2.drawImage(video, 0, 0, canvas2.width, canvas2.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    runModel();
    requestAnimationFrame(drawFrame); 
}

async function loadModel() {
    const session = await ort.InferenceSession.create("yolov8n-seg.onnx");
    console.log("Model loaded!");
    return session;
}

async function loadDeepLab(){

        const modelName = 'pascal';   // set to your preferred model, either `pascal`, `cityscapes` or `ade20k`
        const quantizationBytes = 2;  // either 1, 2 or 4
        return await deeplab.load({base: modelName, quantizationBytes});
}

 function preprocessImage(image) {
//     const canvas = document.createElement("canvas");
//     const ctx = canvas.getContext("2d");
//     canvas.width = 640;
//     canvas.height = 640;
//    // ctx.drawImage(image, 0, 0, 640, 640);

    // Get pixel data
    const imageData = ctx.getImageData(0, 0, canvas.width,  canvas.height);
    const data = imageData.data;
    const floatArray = new Float32Array(3 * 640 * 640);

    for (let i = 0; i < 640 * 640; i++) {
        floatArray[i] = data[i * 4] / 255.0;        // R
        floatArray[i + 640 * 640] = data[i * 4 + 1] / 255.0; // G
        floatArray[i + 2 * 640 * 640] = data[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor("float32", floatArray, [1, 3, 640, 640]);
}

async function runModel() {
    const inputTensor = preprocessImage();

    const feeds = { images: inputTensor };
    const results = await session.run(feeds);

    console.log("Detection output:", results);

}