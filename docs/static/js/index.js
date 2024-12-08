window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	// Check for click events on the navbar burger icon

	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	bulmaSlider.attach();
	document.getElementById('copyButton').addEventListener('click', copyText);

})

const copyText = () => {
	const text = `@article{taomaniskill3,
        title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
        author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
        journal = {arXiv preprint arXiv:2410.00425},
        year={2024},
    }`;

	navigator.clipboard.writeText(text).then(() => {
		const messageDiv = document.getElementById('message');
		messageDiv.textContent = 'Copied to clipboard';
		messageDiv.style.color = 'green';

		setTimeout(() => {
			messageDiv.textContent = '';
		}, 2000);
	}).catch(err => {
		console.error('Failed to copy: ', err);
	});
};

