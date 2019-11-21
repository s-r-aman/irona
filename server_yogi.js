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
const port = process.env.PORT || 5000;

app.post("/predict", (req, res) => {
  const {
    body: { input }
  } = req;

  let options = {
    mode: "text",
    pythonPath: "./pythonenv/bin/python",
    args: [input]
  };

  PythonShell.run("./pythonenv/predict.py", options, function(err, results) {
    if (err) throw err;
    //send response
    const prediction = results;
    res.send({ prediction });
  });
});

app.listen(port, () => console.log(`App is listening on port ${port}.`));
