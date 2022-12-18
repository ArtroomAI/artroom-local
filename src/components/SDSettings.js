import { React, useState, useEffect } from 'react';
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
    createStandaloneToast,
    Text,
    Icon
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import { IoMdCloud } from 'react-icons/io';

function SDSettings () {
    const { ToastContainer, toast } = createStandaloneToast();

    const [width, setWidth] = useRecoilState(atom.widthState);
    const [height, setHeight] = useRecoilState(atom.heightState);
    const [text_prompts, setTextPrompts] = useRecoilState(atom.textPromptsState);
    const [batch_name, setBatchName] = useRecoilState(atom.batchNameState);
    const [steps, setSteps] = useRecoilState(atom.stepsState);
    const [aspect_ratio, setAspectRatio] = useRecoilState(atom.aspectRatioState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(atom.aspectRatioSelectionState);
    const [seed, setSeed] = useRecoilState(atom.seedState);
    const [use_random_seed, setUseRandomSeed] = useRecoilState(atom.useRandomSeedState);
    const [n_iter, setNIter] = useRecoilState(atom.nIterState);
    const [sampler, setSampler] = useRecoilState(atom.samplerState);
    const [cfg_scale, setCFGScale] = useRecoilState(atom.CFGScaleState);
    const [init_image, setInitImage] = useRecoilState(atom.initImageState);
    const [strength, setStrength] = useRecoilState(atom.strengthState);
    const [ckpt, setCkpt] = useRecoilState(atom.ckptState);
    const [ckpts, setCkpts] = useState([]);
    const [cloudMode, setCloudMode] = useRecoilState(atom.cloudModeState);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);

    const getCkpts = () => {
        window.getCkpts(ckpt_dir).then((result) => {
            console.log(result);
            setCkpts(result);
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
        [ckpt_dir]
    );

    const uploadSettings = (event) => {
        window.uploadSettings().then((result) => {
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
                    setTextPrompts(settings.text_prompts);
                    setBatchName(settings.batch_name);
                    setSteps(settings.steps);
                    setAspectRatio(settings.aspect_ratio);
                    setWidth(settings.width);
                    setHeight(settings.height);
                    setSeed(settings.seed);
                    setInitImage(settings.init_image);
                    setCFGScale(settings.cfg_scale);
                    setNIter(settings.n_iter);
                    setSampler(settings.sampler);
                    setStrength(settings.strength);
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
                            onChange={(event) => setBatchName(event.target.value)}
                            value={batch_name}
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
                                onChange={(v) => {
                                    setNIter(v);
                                }}
                                value={n_iter}
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
                                onChange={(v) => {
                                    setSteps(v);
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
                                        if (event.target.value !== 'Custom') {
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
                                        onChange={(event) => setAspectRatio(event.target.value)}
                                        value={aspect_ratio}
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
                                isReadOnly={aspect_ratio === 'Init Image'}
                                max={1920}
                                min={256}
                                name="width"
                                onChange={(v) => {
                                    setWidth(v);
                                }}
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
                                    isOpen={!(aspect_ratio === 'Init Image')}
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
                                isReadOnly={!(aspect_ratio === 'None') || aspect_ratio === 'Init Image'}
                                max={1920}
                                min={256}
                                onChange={(v) => setHeight(v)}
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
                                    isOpen={!(aspect_ratio === 'Init Image')}
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
                            onChange={(v) => {
                                setCFGScale(v);
                            }}
                            value={cfg_scale}
                            variant="outline"
                        >
                            <NumberInputField id="cfg_scale" />
                        </NumberInput>
                    </FormControl>

                    {init_image.length > 0
                        ? <FormControl className="strength-input">
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
                                isDisabled={init_image.length === 0}
                                max={0.99}
                                min={0.0}
                                name="strength"
                                onChange={(v) => {
                                    setStrength(v);
                                }}
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
                                    isOpen={!(init_image.length === 0)}
                                    label={`${strength}`}
                                    placement="right"
                                >
                                    <SliderThumb />
                                </Tooltip>
                            </Slider>
                        </FormControl>
                        : <></>}

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
                                ddim
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpmpp_2m"
                            >
                                dpmpp_2m
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpmpp_2s_ancestral"
                            >
                                dpmpp_2s_ancestral
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="euler"
                            >
                                euler
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="euler_a"
                            >
                                euler_ancestral
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpm_2"
                            >
                                dpm_2
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="dpm_a"
                            >
                                dpm_2_ancestral
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="lms"
                            >
                                lms
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="heun"
                            >
                                heun
                            </option>

                            <option
                                style={{ 'backgroundColor': '#080B16' }}
                                value="plms"
                            >
                                plms
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
                                isDisabled={use_random_seed}
                                min={0}
                                name="seed"
                                onChange={(v) => {
                                    setSeed(v);
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
                                isChecked={use_random_seed}
                                name="use_random_seed"
                                onChange={() => {
                                    setUseRandomSeed(!use_random_seed);
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
