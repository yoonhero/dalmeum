let constraints = { video: { facingMode: "user"}, audio: false};
const cameraView = document.querySelector("#camera--view");
const cameraSensor = document.querySelector("#camera--sensor");
const cameraTrigger = document.querySelector("#camera--trigger");
const restartButton = document.querySelector(".restart")


const resultContainer = document.querySelector("#result")
const cameraOutput = document.querySelector(".camera--output");
const predictionOutput = document.querySelector(".pred--output");
const predictionLabel = document.querySelector(".label")


function cameraStart(){
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream){
            track = stream.getTracks()[0];
            cameraView.srcObject = stream;
        })
        .catch(function(error){
            console.error("카메라에 문제가 있습니다.", error);
        })
}

restartButton.addEventListener("click", () => {
    cameraTrigger.classList.remove("hidden")
    restartButton.classList.add("hidden")
    resultContainer.classList.add("hidden")    
    cameraStart()
})

cameraTrigger.addEventListener("click", function(){
    cameraSensor.width = cameraView.videoWidth; 
    cameraSensor.height = cameraView.videoHeight;
    
    cameraSensor.getContext("2d").drawImage(cameraView, 0, 0);

    // Stop Camera
    cameraView.srcObject = null

    // Toggle button
    cameraTrigger.classList.add("hidden")
    restartButton.classList.remove("hidden")
    
    cameraOutput.src = cameraSensor.toDataURL("image/webp");

    
    axios.post('https://8c4c-119-194-35-226.jp.ngrok.io/predict', {
        image: cameraSensor.toDataURL(),
    },  {
        "Content-Type": "application/json",
    })
    .then(function (response) {
        // console.log(response);///
        resultContainer.classList.remove("hidden")
        cameraSensor.classList.add("hidden")
        predictionOutput.src = response.data.body.url
        // predictionLabel.innerText = response.data.body.player_name
        // predictionLabel.innerTex String((response.data.body.confidence)*100) + "%"
    })
    .catch(function (error) {
        console.log(error);
    });
    // var xhr = new XMLHttpRequest();
    // xhr.withCredentials = true;
    // xhr.open('POST', 'http://localhost:5000/predict', cameraSensor.toDataURL(), true);
    // // Set other request headers and body
    // xhr.send();
//        cameraOutput.classList.add("taken");
});

window.addEventListener("load", cameraStart, false);
axios.defaults.withCredentials = true;