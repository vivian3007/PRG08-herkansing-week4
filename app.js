const video = document.getElementById("video");
const log = document.getElementById("array")

function initApplication() {
    Notification.requestPermission();
    new Notification("Starting the app");

    setupCamera();
    video.addEventListener("play", () => generateLandmarks());
}

//setup the camera
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Webcam not available");
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            facingMode: "user",
            width: 720,
            height: 405
        }
    });

    video.srcObject = stream;
    console.log(video.height);
}

//show when the model is loaded
function modelLoaded() {
    console.log('Model Loaded!');
}

//generate the landmarks on the face
function generateLandmarks() {
    const canvas = faceapi.createCanvasFromMedia(video);
    const videoElement = document.querySelector(".video-wrapper");
    videoElement.append(canvas);
    const displaySize = { width: video.width, height: video.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
        const detections = await faceapi
            .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks();

        if (detections[0]?.landmarks) {
            logData(detections);
        } else {
            console.log("No face found");
        }

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    }, 1000 * 4);
}

//kNear
const k = 3;
const machine = new kNear(k);

let predictionsArray = [];

//button variables
let sad = document.getElementById('sad');
let neutral = document.getElementById('neutral');
let happy = document.getElementById('happy');
let save = document.getElementById('save');

save.addEventListener("click", (event) => saveModel(event));

let buttonsArray = [sad, neutral, happy, save];

//button handlers
function buttonSadHandler(e) {
    e.preventDefault()
    machine.learn(predictionsArray,'sad')
    console.log('sad')
}

function buttonNeutralHandler(e) {
    e.preventDefault()
    machine.learn(predictionsArray, 'neutral')
    console.log('neutral')
}

function buttonHappyHandler(e) {
    e.preventDefault()
    machine.learn(predictionsArray, 'happy')
    console.log('happy')
}

//prediction
function logData(detections) {
    predictionsArray = [];

    console.log(predictionsArray)

    sad.addEventListener("click", (event) => buttonSadHandler(event, detections));
    neutral.addEventListener("click", (event) => buttonNeutralHandler(event, detections));
    happy.addEventListener("click", (event) => buttonHappyHandler(event, detections));

    const mouthCoordinates = detections[0].landmarks.getMouth();

    for (let i = 0; i < mouthCoordinates.length; i++) {
        predictionsArray.push(mouthCoordinates[i]._x);
        predictionsArray.push(mouthCoordinates[i]._y);
    }

    let prediction = machine.classify(predictionsArray);
    console.log(`You look ${prediction}`);

    log.innerHTML = prediction;
}

//save model
function saveModel(e){
    e.preventDefault()
    buttonsArray.forEach(button => {
        button.style.display = 'none';
    });

    const json = JSON.stringify(machine)
    console.log(json)
}

Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri("face-api/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("face-api/models")
]).then(initApplication);
