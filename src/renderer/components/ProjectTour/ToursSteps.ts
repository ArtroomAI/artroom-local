import PaintTutorial from '../../images/painting_tutorial.gif';


const outro = {
    target: '.tour-link',
    content: 'Each page has its own tutorial, so if you ever feel confused or want to see a tutorial again, click on this button! Enjoy!'
};


export const TOUR_STEPS_MAIN = [{
    target: '.image-box',
    content: 'Welcome to Artroom! This is an app that lets you generate AI Art',
    disableBeacon: true // This makes the tour to start automatically without clicking
}, {
    target: '.text-prompts',
    content:
        'In this box, you can type in what you want to generate and...'
}, {
    target: '.run-button',
    content: 'then just press run! It\'s that easy! We\'ll go over more advanced settings '
}, {
    target: '.folder-name-input',
    content: 'This is the folder name your images will go to.'
}, {
    target: '.settings-nav',
    content: 'The overall destination can be changed in settings. More on that later!'
}, {
    target: '.num-images-input',
    content: 'You can set how many images you want to generate'
}, {
    target: '.size-input',
    content: 'And how big you want them to be.Please note that increasing your image size exponentially increases generation time. A good alternative is using an upscaler, which is its own menu on the left.'
}, {
    target: '.steps-input',
    content: 'Steps determine how long you want the model to spend on generating your image. The more steps you have, the longer it will take but you\'ll get better results. The results are less impactful the more steps you have, so you may stop seeing improvement after 100 steps. 50 is typically a good number.'
}, {
    target: '.cfg-scale-input',
    content: 'CFG Scale determines how intense the generations are. A typical value is around 5-15 with higher numbers telling the AI to stay closer to the prompt you typed'
}, {
    target: '.samplers-input',
    content: 'Samplers determine how the AI model goes about the generation. Each sampler has its own aesthetic (sometimes they may even end up with the same results). Play around with them and see which ones you prefer!'
}, {
    target: '.model-ckpt-input',
    content: 'Here you\'ll have a list of your models. There are tons to download and each one is good at their own thing. Try some out! You can find a bunch in the resources in our discord page. PLEASE EXERCISE CAUTION WHEN DOWNLOADING FILES FROM THE INTERNET'
}, {
    target: '.seed-input',
    content: 'Seed controls your randomness. If you pick the same seed with the same settings, you\'ll get the same results every time. Note: it may not work across devices'
}, {
    target: '.load-settings-button',
    content: 'Your settings will be saved on each run. You can click here to upload one of your favorite settings.json files and load them into the Artroom'
}, {
    target: '.starting-image',
    content: 'If you have an idea of an image you want to use as your base, click on the Upload button OR right click one of the generated images and build off of your creation!'
}, {
    target: '.negative-prompts',
    content: 'You can even write what you DON\'T want to show up. A good default is: text, cropped, jpeg artifacts, signature, watermark, blurry'
}, outro];
export const TOUR_STEPS_PAINT = [{
    target: '.paint-output',
    content: 'This one gives you a bit more control over your creations. You can use a Paint tool to either PAINT on your image or create a MASK',
    disableBeacon: true // This makes the tour to start automatically without clicking
}, {
    target: '.starting-image',
    content:
          'First upload your starting image'
}, {
    target: '.sd-settings',
    content: 'Notice how all of the settings are exactly the same, although once you upload your image, you\'ll see a "Strength" which will let you determine how strongly you want it to change the output. 0 means your image won\'t change and 1 means it won\'t even look at your original input. Play around with the settings, you\'ll probably want something in betewen.'
}, {
    target: '.paint-output',
    content: 'Then draw on your image here. You can use any color EXCEPT White for the Mask tool and any color if you\'re just painting on your image',
    styles: { options: {
        backgroundImage: { PaintTutorial }
    } }
}, {
    target: '.paint-type',
    content: 'You can select what kind of paint type you want. "Paint" means use the colors as if they were part of the image (like drawing in MSPaint and putting it as your image). "Mask" means ONLY change the parts that is painted over (All colors will be treated the same, EXCEPT you CANNOT use white). "Reverse Mask" means KEEP the painted over section and change EVERYTHING ELSE. Note: If you\'re using queue it currently cannot change your mask for each gen, only settings. This will be included in a later patch.'
}, {
    target: '.run-button',
    content: 'And then just press run! However, outputs won\'t show up in a preview this time'
}, {
    target: '.viewer-link',
    content: 'You can view your images by clicking on the view button, which will take you to your folders'
}, outro];

export const TOUR_STEPS_QUEUE = [{
    target: '.queue',
    content: 'This is the Queue. You can start up a bunch of gens all at once and come back to see your final results.',
    disableBeacon: true // This makes the tour to start automatically without clicking
}, {
    target: '.start-stop-button',
    content: 'The queue won\'t be active until you click on Start. You can always turn off the queue by clicking the Stop button (which will appear only when queue is active). NOTE: If for some reason the queue gets glitched out, try clearing the queue and then stopping and restarting the queue. '
}, {
    target: '.queue-table',
    content: 'When you add an image to your queue, you\'ll see your lineup here. You can delete and see the settings you put in your queue.'
}, {
    target: '.clear-queue-button',
    content: 'Don\'t forget to periodically clear out your queue of stuff you don\'t want anymore.'
}, outro];

export const TOUR_STEPS_UPSCALE = [{
    target: '.upscale',
    content: 'This is where you can upscale your images and correct imperfections on faces',
    disableBeacon: true // This makes the tour to start automatically without clicking
}, {
    target: '.upscale-images-input',
    content: 'Choose which images you want to upscale. You can select ALL the images in the folder if you want to. Don\'t worry about accidentally selecting a folder when you highlight, it\'ll filter only images automatically.'
}, {
    target: '.upscale-dest-input',
    content: 'Choose where you want the images to go. If you leave it blank, the upscaled images will go to the same folder where your images came from.'
}, {
    target: '.upscaler-input',
    content: 'Choose what kind of upscaler you want to use. GFPGAN, CodeFormer, and RestoreFormer all fix faces while RealESRGAN just makes the image bigger. If you ONLY want to fix the face without making it bigger, you can set the multipler to 1'
}, {
    target: '.upscale-factor-input',
    content: 'Enter a whole number (1,2,3) for how much you want to multiple your size by. For example, if you choose 2, then your 512x512 image will become 1024x1024'
}, {
    target: '.upscale-button',
    content: 'When you\'re ready, click Upscale and you\'ll have a black command prompt window show you your progress. Enjoy!'
}, outro];

export const TOUR_STEPS_SETTINGS = [{
    target: '.settings',
    content: 'This is the Settings, where you can control the backend parts of Artroom.',
    disableBeacon: true // This makes the tour to start automatically without clicking
}, {
    target: '.image-save-path-input',
    content: 'This is where you can choose where your images go.'
}, {
    target: '.model-ckpt-dir-input',
    content: 'You can save your model ckpts in any drive or folder you\'d like. Just make sure it\'s an actual folder (So do E:/model_weights instead of just E:/)'
},
{
    target: '.highres-fix-input',
    content: 'If you turn on highres fix, once you get past 1024x1024, Artroom will generate a smaller image, upscale, and then img2img to improve quality and avoid weirdness that comes from bigger generations'
}, {
    target: '.long-save-path-input',
    content: 'If you want to further break down your images in the folder, you can save each generation into separate folders based on the prompt used'
}, {
    target: '.save-grid-input',
    content: 'When you finish a batch, you can save a grid of the images for easy viewing'
}, {
    target: '.debug-mode-input',
    content: 'When Debug Mode is set, a terminal window will appear with detailed python output during image generation. Toggling this box will restart Python process with terminal visible or hidden.'
}, {
    target: '.save-settings-button',
    content: 'Don\'t forget to click save!'
}, outro];
