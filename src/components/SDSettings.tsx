import React from 'react';
import { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    Box,
    Button,
    FormControl,
    FormLabel,
    Input,
    VStack,
    HStack,
    NumberInput,
    NumberInputField,
    Slider,
    SliderTrack,
    SliderFilledTrack,
    SliderThumb,
    Tooltip,
    Checkbox,
    Select,
    Spacer,
    Text,
    Icon,
    useToast
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import { IoMdCloud } from 'react-icons/io';

function SDSettings () {
    const toast = useToast({});
    
    const [imageSettings, setImageSettings] = useRecoilState(atom.imageSettingsState)
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(atom.aspectRatioSelectionState);
    const [ckpts, setCkpts] = useState([]);
    const [vaes, setVaes] = useState([]);
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);

    const getCkpts = () => {
        window.api.getCkpts(imageSettings.ckpt_dir).then((result) => {
            // console.log(result);
            setCkpts(result);
        });
        window.api.getVaes(imageSettings.ckpt_dir).then((result) => {
            // console.log(result);
            setVaes(result);
        });
    };

    useEffect(
        () => {
            getCkpts();
        },
        []
    );

    useEffect(
        () => {
            getCkpts();
        },
        [imageSettings.ckpt_dir]
    );

    useEffect(
        () => {
            if (imageSettings.width > 0) {
                let newHeight = imageSettings.height;
                if (aspectRatioSelection !== 'Init Image' && aspectRatioSelection !== 'None') {
                    try {
                        const values = imageSettings.aspect_ratio.split(':');
                        const widthRatio = parseFloat(values[0]);
                        const heightRatio = parseFloat(values[1]);
                        if (!isNaN(widthRatio) && !isNaN(heightRatio)) {
                            newHeight = Math.min(
                                1920,
                                Math.floor(imageSettings.width * heightRatio / widthRatio / 64) * 64
                            );
                        }
                    } catch {

                    }
                    setImageSettings({...imageSettings, height: newHeight});
                }
            }
        },
        [imageSettings.width, imageSettings.aspect_ratio]
    );

    const uploadSettings = () => {
        window.api.uploadSettings().then((result) => {
            if (!(result === '')) {
                try {
                    const data = JSON.parse(result);
                    let settings = data;
                    if ('Settings' in data) {
                        settings = data.Settings;
                    }
                    if (!('text_prompts' in settings)) {
                        throw 'Invalid JSON';
                    }
                    console.log(settings);
                    setImageSettings({
                        text_prompts: settings.text_prompts,
                        negative_prompts: settings.negative_prompts,
                        batch_name: settings.batch_name,
                        n_iter: settings.n_iter,
                        steps: settings.steps,
                        strength: settings.strength,
                        cfg_scale: settings.cfg_scale,
                        sampler: settings.sampler,
                        width: settings.width,
                        height: settings.height,
                        aspect_ratio: settings.aspect_ratio,
                        ckpt: settings.ckpt,
                        vae: settings.vae,
                        seed: settings.seed,
                        speed: settings.speed,
                        save_grid: settings.save_grid,
                        use_random_seed: settings.use_random_seed,
                        init_image: settings.init_image,
                        mask_image: '',
                        invert: false,
                        image_save_path: settings.image_save_path,
                        ckpt_dir: settings.ckpt_dir,
                    })
                } catch (err) {
                    toast({
                        'title': 'Load Failed',
                        'status': 'error',
                        'position': 'top',
                        'duration': 3000,
                        'isClosable': false,
                        'containerStyle': {
                            'pointerEvents': 'none'
                        }
                    });
                }
            }
        });
    };

    return (
        <Flex
            pr="10"
            width="300px"
        >
            <Box
                m="-1"
                p={4}
                rounded="md"
            >
                <VStack
                    className="sd-settings"
                    spacing={3}
                >
                    <FormControl className="folder-name-input">
                        <FormLabel htmlFor="batch_name">
                            Output Folder
                        </FormLabel>

                        <Input
                            id="batch_name"
                            name="batch_name"
                            onChange={(event) => setImageSettings({...imageSettings, batch_name: event.target.value})}
                            value={imageSettings.batch_name}
                            variant="outline"
                        />
                    </FormControl>

                    <HStack>
                        <FormControl className="num-images-input">
                            <FormLabel htmlFor="n_iter">
                                № of Images
                            </FormLabel>

                            <NumberInput
                                id="n_iter"
                                min={1}
                                name="n_iter"
                                onChange={(v, n) => {
                                    setImageSettings({...imageSettings, n_iter: n});
                                }}
                                value={imageSettings.n_iter}
                                variant="outline"
                            >
                                <NumberInputField id="n_iter" />
                            </NumberInput>
                        </FormControl>

                        <FormControl className="steps-input">
                            <HStack>
                                <FormLabel htmlFor="steps">
                                    № of Steps
                                </FormLabel>

                                <Tooltip
                                    fontSize="md"
                                    label="Steps determine how long you want the model to spend on generating your image. The more steps you have, the longer it will take but you'll get better results. The results are less impactful the more steps you have, so you may stop seeing improvement after 100 steps. 50 is typically a good number"
                                    placement="left"
                                    shouldWrapChildren
                                >
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>
                            </HStack>

                            <NumberInput
                                id="steps"
                                min={1}
                                name="steps"
                                onChange={(v, n) => {
                                    setImageSettings({...imageSettings, steps: n});
                                }}
                                value={imageSettings.steps}
                                variant="outline"
                            >
                                <NumberInputField id="steps" />
                            </NumberInput>
                        </FormControl>
                    </HStack>

                    <Box
                        className="size-input"
                        width="100%"
                    >
                        <FormControl
                            className=" aspect-ratio-input"
                            marginBottom={2}
                        >
                            <FormLabel htmlFor="AspectRatio">
                                Fixed Aspect Ratio
                            </FormLabel>

                            <HStack>
                                <Select
                                    id="aspect_ratio_selection"
                                    name="aspect_ratio_selection"
                                    onChange={(event) => {
                                        setAspectRatioSelection(event.target.value);
                                        if (event.target.value !== 'Custom') {
                                            setImageSettings({...imageSettings, aspect_ratio: event.target.value});
                                        }
                                    }}
                                    value={aspectRatioSelection}
                                    variant="outline"
                                >
                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="None"
                                    >
                                        None
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="Init Image"
                                    >
                                        Init Image
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="1:1"
                                    >
                                        1:1
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="1:2"
                                    >
                                        1:2
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="2:1"
                                    >
                                        2:1
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="4:3"
                                    >
                                        4:3
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="3:4"
                                    >
                                        3:4
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="16:9"
                                    >
                                        16:9
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="9:16"
                                    >
                                        9:16
                                    </option>

                                    <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="Custom"
                                    >
                                        Custom
                                    </option>
                                </Select>

                                {aspectRatioSelection === 'Custom'
                                    ? <Input
                                        id="aspect_ratio"
                                        name="aspect_ratio"
                                        onChange={(event) =>                                             setImageSettings({...imageSettings, aspect_ratio: event.target.value})}
                                        value={imageSettings.aspect_ratio}
                                        variant="outline"
                                    />
                                    : <></>}

                            </HStack>
                        </FormControl>

                        <FormControl className="width-input">
                            <FormLabel htmlFor="Width">
                                Width:
                            </FormLabel>

                            <Slider
                                colorScheme="teal"
                                defaultValue={512}
                                id="width"
                                isReadOnly={imageSettings.aspect_ratio === 'Init Image'}
                                max={1920}
                                min={256}
                                name="width"
                                onChange={(v) => {
                                    setImageSettings({...imageSettings, width: v});
                                }}                                
                                step={64}
                                value={imageSettings.width}
                                variant="outline"
                            >
                                <SliderTrack bg="#EEEEEE">
                                    <Box
                                        position="relative"
                                        right={10}
                                    />

                                    <SliderFilledTrack bg="#4f8ff8" />
                                </SliderTrack>

                                <Tooltip
                                    bg="#4f8ff8"
                                    color="white"
                                    isOpen={!(imageSettings.aspect_ratio === 'Init Image')}
                                    label={`${imageSettings.width}`}
                                    placement="right"
                                >
                                    <SliderThumb />
                                </Tooltip>
                            </Slider>
                        </FormControl>

                        <FormControl className="height-input">
                            <FormLabel htmlFor="Height">
                                Height:
                            </FormLabel>

                            <Slider
                                defaultValue={512}
                                isReadOnly={imageSettings.aspect_ratio === 'Init Image'}
                                max={1920}
                                min={256}
                                onChange={(v) => {
                                    setImageSettings({...imageSettings, height: v});
                                }}                                        
                                step={64}
                                value={imageSettings.height}
                            >
                                <SliderTrack bg="#EEEEEE">
                                    <Box
                                        position="relative"
                                        right={10}
                                    />

                                    <SliderFilledTrack bg="#4f8ff8" />
                                </SliderTrack>

                                <Tooltip
                                    bg="#4f8ff8"
                                    color="white"
                                    isOpen={!(imageSettings.aspect_ratio === 'Init Image')}
                                    label={`${imageSettings.height}`}
                                    placement="right"
                                >
                                    <SliderThumb />
                                </Tooltip>
                            </Slider>
                        </FormControl>
                    </Box>

                    <FormControl className="cfg-scale-input">
                        <HStack>
                            <FormLabel htmlFor="cfg_scale">
                                Prompt Strength (CFG):
                            </FormLabel>

                            <Spacer />

                            <Tooltip
                                fontSize="md"
                                label="Prompt Strength or CFG Scale determines how intense the generations are. A typical value is around 5-15 with higher numbers telling the AI to stay closer to the prompt you typed"
                                placement="left"
                                shouldWrapChildren
                            >
                                <FaQuestionCircle color="#777" />
                            </Tooltip>
                        </HStack>

                        <NumberInput
                            id="cfg_scale"
                            min={0}
                            name="cfg_scale"
                            onChange={(v, n) => {
                                setImageSettings({...imageSettings, cfg_scale: n});
                            }}         
                            value={imageSettings.cfg_scale}
                               variant="outline"
                        >
                            <NumberInputField id="cfg_scale" />
                        </NumberInput>
                    </FormControl>

                    <FormControl className="strength-input">
                            <HStack>
                                <FormLabel htmlFor="Strength">
                                    Image Variation Strength:
                                </FormLabel>
                                <Spacer />
                                <Tooltip
                                    fontSize="md"
                                    label="Strength determines how much your output will resemble your input image. Closer to 0 means it will look more like the original and closer to 1 means use more noise and make it look less like the input"
                                    placement="left"
                                    shouldWrapChildren
                                >
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>
                            </HStack>

                            <Slider
                                defaultValue={0.75}
                                id="strength"
                                max={0.99}
                                min={0.0}
                                name="strength"
                                onChange={(v) => {
                                    setImageSettings({...imageSettings, strength: v});
                                }}        
                                step={0.01}
                                value={imageSettings.strength}
                                variant="outline"
                            >
                                <SliderTrack bg="#EEEEEE">
                                    <Box
                                        position="relative"
                                        right={10}
                                    />

                                    <SliderFilledTrack bg="#4f8ff8" />
                                </SliderTrack>

                                <Tooltip
                                    bg="#4f8ff8"
                                    color="white"
                                    isOpen={true}
                                    label={`${imageSettings.strength}`}
                                    placement="right"
                                >
                                    <SliderThumb />
                                </Tooltip>
                            </Slider>
                    </FormControl>

                    <FormControl className="samplers-input">
                        <HStack>
                            <FormLabel htmlFor="Sampler">
                                Sampler
                            </FormLabel>

                            <Spacer />

                            <Tooltip
                                fontSize="md"
                                label="Samplers determine how the AI model goes about the generation. Each sampler has its own aesthetic (sometimes they may even end up with the same results). Play around with them and see which ones you prefer!"
                                placement="left"
                                shouldWrapChildren
                            >
                                <FaQuestionCircle color="#777" />
                            </Tooltip>

                        </HStack>

                        <Select
                            id="sampler"
                            name="sampler"
                            onChange={(event) => setImageSettings({...imageSettings, sampler: event.target.value})}
                            value={imageSettings.sampler}
                            variant="outline"
                        >
                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="ddim"
                            >
                                DDIM
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpmpp_2m"
                            >
                                DPM++ 2M Karras
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpmpp_2s_ancestral"
                            >
                                DPM++ 2S Ancestral Karras
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="euler"
                            >
                                Euler
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="euler_a"
                            >
                                Euler Ancestral
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpm_2"
                            >
                                DPM 2
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpm_a"
                            >
                                DPM 2 Ancestral
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="lms"
                            >
                                LMS
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="heun"
                            >
                                Heun
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="plms"
                            >
                                PLMS
                            </option>
                        </Select>
                    </FormControl>

                    <FormControl className="model-ckpt-input">
                        <FormLabel htmlFor="Ckpt">
                            <HStack>
                                <Text>
                                    Model
                                </Text>

                                {cloudMode
                                    ? <Icon as={IoMdCloud} />
                                    : null}
                            </HStack>
                        </FormLabel>

                        <Select
                            id="ckpt"
                            name="ckpt"
                            onChange={(event) => setImageSettings({...imageSettings, ckpt: event.target.value})}
                            onMouseEnter={getCkpts}
                            value={imageSettings.ckpt}
                            variant="outline"
                        >
                            {ckpts.length > 0
                                ? <option
                                    style={{ 'backgroundColor': '#080B16' }}
                                    value=""
                                >
                                    Choose Your Model Weights
                                </option>
                                : <></>}

                            {ckpts.map((ckpt_option, i) => (<option
                                key={i}
                                style={{ 'backgroundColor': '#080B16' }}
                                value={ckpt_option}
                            >
                                {ckpt_option}
                            </option>))}
                        </Select>
                    </FormControl>

                    <FormControl className="vae-ckpt-input">
                        <FormLabel htmlFor="Vae">
                            <HStack>
                                <Text>
                                    VAE
                                </Text>

                                {cloudMode
                                    ? <Icon as={IoMdCloud} />
                                    : null}
                            </HStack>
                        </FormLabel>

                        <Select
                            id="vae"
                            name="vae"
                            onChange={(event) => setImageSettings({...imageSettings, vae: event.target.value})}
                            onMouseEnter={getCkpts}
                            value={imageSettings.vae}
                            variant="outline"
                        >
                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value=""
                            >
                                No vae
                            </option>
                            {vaes.map((ckpt_option, i) => (<option
                                key={i}
                                style={{ 'backgroundColor': '#080B16' }}
                                value={ckpt_option}
                            >
                                {ckpt_option}
                            </option>))}
                        </Select>
                    </FormControl>

                    <HStack className="seed-input">
                        <FormControl>
                            <HStack>
                                <FormLabel htmlFor="seed">
                                    Seed:
                                </FormLabel>

                                <Spacer />

                                <Tooltip
                                    fontSize="md"
                                    label="Seed controls randomness. If you set the same seed each time and use the same settings, then you will get the same results"
                                    placement="left"
                                    shouldWrapChildren
                                >
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>
                            </HStack>

                            <NumberInput
                                id="seed"
                                isDisabled={imageSettings.use_random_seed}
                                min={0}
                                name="seed"
                                onChange={(v) => {
                                    setImageSettings({...imageSettings, seed: parseInt(v)});
                                }}        
                                value={imageSettings.seed}
                                variant="outline"
                            >
                                <NumberInputField id="seed" />
                            </NumberInput>
                        </FormControl>

                        <VStack
                            align="center"
                            justify="center"
                        >
                            <FormLabel
                                htmlFor="use_random_seed"
                                pb="3px"
                            >
                                Random
                            </FormLabel>

                            <Checkbox
                                id="use_random_seed"
                                isChecked={imageSettings.use_random_seed}
                                name="use_random_seed"
                                onChange={() => {
                                    setImageSettings({...imageSettings, use_random_seed: !imageSettings.use_random_seed});
                                }}
                                pb="12px"
                            />
                        </VStack>
                    </HStack>

                    <Button
                        className="load-settings-button"
                        onClick={uploadSettings}
                        w="250px"
                    >
                        Load Settings
                    </Button>
                </VStack>
            </Box>
        </Flex>
    );
}
export default SDSettings;
