let xs = [];
let ys = [];

let m, b;

const learningRate = 0.3;
const optimizer = tf.train.sgd(learningRate);

function setup() {
	createCanvas(400, 400);
	background(0);
	
	m = tf.variable(tf.scalar(0));
	b = tf.variable(tf.scalar(0.5));
}

//Given x-values, return predicted y-values
function predict(x) {
	const x_t = tf.tensor1d(x);
	return x_t.mul(m).add(b);
}

//Mean-squared loss function to be minimized by sgd optimization
function loss(pred, labels) {
	const y_t = tf.tensor1d(labels);
	return pred.sub(y_t).square().mean();
}

function mousePressed() {
	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, 0, height, 1, 0);
	xs.push(x);
	ys.push(y);
}

function draw() {
	background(0);
	stroke(255);
	strokeWeight(4);
	
	tf.tidy(() => {
		if(xs.length > 0) {
			optimizer.minimize(function() {
				return loss(predict(xs), ys);
			});
		}});
	
	for (let i = 0; i < xs.length; i++) {
		let px = map(xs[i], 0, 1, 0, width);
		let py = map(ys[i], 0, 1, height, 0);
		point(px, py);
	}
	
	let lineX = [0, 1];
	let yPred = tf.tidy(() => predict(lineX));
	let lineY = yPred.dataSync();
	yPred.dispose();
	
	let x1 = map(lineX[0], 0, 1, 0, width);
	let x2 = map(lineX[1], 0, 1, 0, width);
	let y1 = map(lineY[0], 0, 1, height, 0);
	let y2 = map(lineY[1], 0, 1, height, 0);
	
	line(x1, y1, x2, y2);
	
	//console.log(tf.memory().numTensors);
	
}