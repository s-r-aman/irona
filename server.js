const { PythonShell } = require("python-shell");
const express = require("express");
const fileUpload = require("express-fileupload");
const cors = require("cors");
const bodyParser = require("body-parser");
const morgan = require("morgan");
const _ = require("lodash");

const app = express();
// enable files upload
app.use(
  fileUpload({
    createParentPath: true
  })
);

//add other middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan("dev"));

//start app
const port = process.env.PORT || 3000;

app.post("/upload", async (req, res) => {
  try {
    if (!req.files) {
      res.send({
        status: false,
        message: "No file uploaded"
      });
    } else {
      //Use the name of the input field (i.e. "image") to retrieve the uploaded file
      let image = req.files.image;
      if (!["image/jpeg", "image/png", "image/jpg"].includes(image.mimetype)) {
        res
          .status(404)
          .send({ err: "Please send only .png|.jpeg|.jpg files only!" });
      }
      //Use the mv() method to place the file in upload directory (i.e. "uploads")
      image.mv("./uploads/" + image.name);

      let options = {
        mode: "text",
        pythonPath: "./pythonenv/bin/python",
        // pythonOptions: ["-i"], // get print results in real-time
        // scriptPath: "./pythonenv/sample_request",
        args: [`./uploads/${image.name}`]
      };

      PythonShell.run("./pythonenv/sample_request.py", options, function(
        err,
        results
      ) {
        if (err) throw err;
        //send response
        const {
          predictions: [predictions]
        } = JSON.parse(results);
        res.send({
          isCursed: predictions[1] > 0.5,
          predictions
        });
      });
    }
  } catch (err) {
    res.status(500).send(err);
  }
});

app.listen(port, () => console.log(`App is listening on port ${port}.`));
