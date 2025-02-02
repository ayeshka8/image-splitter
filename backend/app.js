const express = require("express");
const multer = require("multer");
const path = require("path");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.static("uploads"));

const storage = multer.diskStorage({
    destination: "./uploads/",
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage });

app.post("/upload", upload.single("image"), async (req, res) => {
    if (!req.file) return res.status(400).json({ message: "No file uploaded" });

    try {
        // Send image to Flask for processing
        const flaskResponse = await axios.post("http://127.0.0.1:5001/process", { //const flaskResponse = await axios.post("http://image-processor:5001/process", { //edited for docker runs

            image_path: req.file.filename
        });

        res.json({ 
            message: "Processing complete!", 
            images: flaskResponse.data.images 
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "Image processing failed!" });
    }
});

app.listen(port, () => console.log(`Server running at http://localhost:${port}`));
