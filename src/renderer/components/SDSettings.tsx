import React, { useCallback } from 'react';
import { useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
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
import { aspectRatioState, batchNameState, cfgState, ckptState, heightState, initImageState, iterationsState, loadSettingsFromFile, modelsDirState, randomSeedState, samplerState, seedState, stepsState, strengthState, vaeState, widthState } from '../SettingsManager';

function SDSettings () {
    const toast = useToast({});

    const cloudMode = useRecoilValue(atom.cloudModeState);

    const modelDirs = useRecoilValue(modelsDirState);
    const [ckpts, setCkpts] = useState([]);
    const [vaes, setVaes] = useState([]);

    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(atom.aspectRatioSelectionState);

    const [width, setWidth] = useRecoilState(widthState);
    const [height, setHeight] = useRecoilState(heightState);
    const [aspectRatio, setAspectRatio] = useRecoilState(aspectRatioState);
    const [batchName, setBatchName] = useRecoilState(batchNameState);
    const [iterations, setIterations] = useRecoilState(iterationsState);
    const [steps, setSteps] = useRecoilState(stepsState);
    const [cfg, setCfg] = useRecoilState(cfgState);
    const [sampler, setSampler] = useRecoilState(samplerState);
    const [strength, setStrength] = useRecoilState(strengthState);
    const [seed, setSeed] = useRecoilState(seedState);
    const initImage = useRecoilValue(initImageState);
    const [ckpt, setCkpt] = useRecoilState(ckptState);
    const [vae, setVae] = useRecoilState(vaeState);
    const [randomSeed, setRandomSeed] = useRecoilState(randomSeedState);

    const getCkpts = useCallback(() => {
        window.api.getCkpts(modelDirs).then(setCkpts);
        window.api.getVaes(modelDirs).then(setVaes);
    }, [modelDirs]);

    useEffect(() => {
        getCkpts();
    }, []);

    useEffect(() => {
        getCkpts();
    }, [getCkpts]);

    useEffect(
        () => {
            if (width > 0) {
                let newHeight = height;
                if (aspectRatioSelection !== 'Init Image' && aspectRatioSelection !== 'None') {
                    try {
                        const values = aspectRatio.split(':');
                        const widthRatio = parseFloat(values[0]);
                        const heightRatio = parseFloat(values[1]);
                        if (!isNaN(widthRatio) && !isNaN(heightRatio)) {
                            newHeight = Math.min(
                                1920,
                                Math.floor(width * heightRatio / widthRatio / 64) * 64
                            );
                        }
                    } catch {

                    }
                    setHeight(newHeight);
                }
            }
        },
        [width, aspectRatio]
    );

    const uploadSettings = () => {
        window.api.uploadSettings().then((result) => {
            if (result) {
                loadSettingsFromFile(result);
            } else {
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
                            onChange={(event) => setBatchName(event.target.value)}
                            value={batchName}
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
                                    setIterations(isNaN(n) ? 1 : n);
                                }}
                                value={iterations}
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
                                    setSteps(isNaN(n) ? 1 : n);
                                }}
                                value={steps}
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
                                        if (event.target.value === 'Init Image' && !initImage) {
                                            //Switch to aspect ratio to none and print warning that no init image is set
                                            setAspectRatioSelection('None');
                                            setAspectRatio('None');
                                            toast({
                                                'title': 'Invalid Aspect Ratio Selection',
                                                'description': 'Must upload Starting Image first to use its resolution',
                                                'status': 'error',
                                                'position': 'top',
                                                'duration': 3000,
                                                'isClosable': true,
                                                'containerStyle': {
                                                    'pointerEvents': 'none'
                                                }
                                            });
                                        } else if (event.target.value !== 'Custom') {
                                            setAspectRatio(event.target.value);
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

                                    {initImage.length && <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value="Init Image"
                                    >
                                        Init Image
                                    </option>}

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
                                        onChange={(event) => setAspectRatio(event.target.value)}
                                        value={aspectRatio}
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
                                isReadOnly={aspectRatio === 'Init Image'}
                                max={2048}
                                min={256}
                                name="width"
                                onChange={setWidth}                                
                                step={64}
                                value={width}
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
                                    isOpen={!(aspectRatio === 'Init Image')}
                                    label={`${width}`}
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
                                isReadOnly={aspectRatio === 'Init Image'}
                                max={2048}
                                min={256}
                                onChange={setHeight}                                        
                                step={64}
                                value={height}
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
                                    isOpen={!(aspectRatio === 'Init Image')}
                                    label={`${height}`}
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
                                setCfg(isNaN(n) ? 0 : n);
                            }}         
                            value={cfg}
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
                            onChange={setStrength}        
                            step={0.01}
                            value={strength}
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
                                label={`${strength}`}
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
                            onChange={(event) => setSampler(event.target.value)}
                            value={sampler}
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
                            onChange={(event) => setCkpt(event.target.value)}
                            onMouseEnter={getCkpts}
                            value={ckpt}
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
                            onChange={(event) => setVae(event.target.value)}
                            onMouseEnter={getCkpts}
                            value={vae}
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
                                isDisabled={randomSeed}
                                min={0}
                                name="seed"
                                onChange={(v, n) => {
                                    setSeed(isNaN(n) ? 0 : n);
                                }}        
                                value={seed}
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
                                isChecked={randomSeed}
                                name="use_random_seed"
                                onChange={() => {
                                    setRandomSeed((useRandomSeed) => !useRandomSeed);
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
