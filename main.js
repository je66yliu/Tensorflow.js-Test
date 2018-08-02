let xs = [];
let ys = [];
let m, b;


let resetFlag = false;
//Reset button event handler
let resetBtn = document.querySelector('.reset');
//Learning Rate
let rangeSlider = document.querySelector('.stroke');

let optimizer = tf.train.sgd(0.1);

resetBtn.addEventListener('click', function() {
	xs = [];
	ys = [];
	resetFlag = true;
	// m = tf.variable(tf.scalar(0));
	// b = tf.variable(tf.scalar(0.5));
});

//Initialize canvas
function setup() {
	let canvas = createCanvas(400, 400);
	background(0);
	canvas.parent('sketch-holder');
	canvas.mousePressed(() => {
			let x = map(mouseX, 0, width, 0, 1);
			let y = map(mouseY, 0, height, 1, 0);
			xs.push(x);
			ys.push(y);
		}
	);

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



function draw() {
	background(0);
	stroke(255);
	let strk = map(rangeSlider.value, 0, 100, 1, 10);
	strokeWeight(strk);
	document.querySelector('.strokewt').textContent = Math.round(strk);

	//Train linear regression
	tf.tidy(() => {
		if((xs.length > 0) && (ys.length > 0)) {
			optimizer.minimize(function() {
				return loss(predict(xs), ys);
			});
		}});
	//Draw points
	for (let i = 0; i < xs.length; i++) {
		let px = map(xs[i], 0, 1, 0, width);
		let py = map(ys[i], 0, 1, height, 0);
		point(px, py);
	}

	//Draw line
	let lineX = [0, 1];
	let yPred = tf.tidy(() => predict(lineX));
	let lineY = yPred.dataSync();
	yPred.dispose();

	let x1 = map(lineX[0], 0, 1, 0, width);
	let x2 = map(lineX[1], 0, 1, 0, width);
	let y1 = map(lineY[0], 0, 1, height, 0);
	let y2 = map(lineY[1], 0, 1, height, 0);

	line(x1, y1, x2, y2);
	if (resetFlag == true) {
		tf.tidy(() => {
			if ((Math.abs(m.dataSync()[0]) >= 0.001) || (Math.abs(b.dataSync()[0] - 0.5) >= 0.02)) {
				if (m.dataSync()[0] < 0.0) {
					m.assign(m.div(2));
				}
				else {
					m.assign(m.div(2));
				}
				if (b.dataSync()[0] < 0.5) {
					b.assign(b.add(0.015));
					//console.log(b.dataSync());
				}
				else {
					b.assign(b.sub(0.015));
					//console.log(b.dataSync());
				}
			}
			else {
				resetFlag = false;
			}
		});
	}
	//console.log(m.dataSync());
	//console.log(tf.memory().numTensors);
}
