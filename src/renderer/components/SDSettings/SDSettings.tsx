import React, { useCallback } from 'react';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import {
    Flex,
    Box,
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
    Button,
    IconButton,
    useToast,
    ButtonGroup
} from '@chakra-ui/react';
import { FaClipboardList, FaFolder, FaQuestionCircle } from 'react-icons/fa';
import { IoMdCloud } from 'react-icons/io';
import {
    batchNameState,
    cfgState,
    ckptState,
    clipSkipState,
    controlnetState,
    iterationsState,
    checkSettings,
    loraState,
    modelsDirState,
    exifDataSelector,
    randomSeedState,
    removeBackgroundState,
    samplerState,
    seedState,
    stepsState,
    strengthState,
    useRemovedBackgroundState,
    vaeState,
    modelsState,
    CheckSettingsType
} from '../../SettingsManager';
import LoraSelector from './Lora/LoraSelector';
import { AspectRatio } from './AspectRatio';
import Controlnet from './Controlnet/Controlnet';
import RemoveBackground from './RemoveBackground/RemoveBackground';
import { SDSettingsAccordion } from './SDSettingsAccordion';
import { getExifData } from '../../../main/utils/exifData';
import HighresUpscale from './HighresUpscale/HighresUpscale';

type SDSettingsTab = 'default' | 'paint';

function SDSettings ({ tab } : { tab: SDSettingsTab }) {
    const toast = useToast({});
    const cloudMode = useRecoilValue(atom.cloudModeState);

    const modelDirs = useRecoilValue(modelsDirState);
    const models = useRecoilValue(modelsState);

    const [batchName, setBatchName] = useRecoilState(batchNameState);
    const [iterations, setIterations] = useRecoilState(iterationsState);
    const [steps, setSteps] = useRecoilState(stepsState);
    const [cfg, setCfg] = useRecoilState(cfgState);
    const [clipSkip, setClipSkip] = useRecoilState(clipSkipState);

    const [sampler, setSampler] = useRecoilState(samplerState);
    const controlnet = useRecoilValue(controlnetState);
    const [strength, setStrength] = useRecoilState(strengthState);
    const [seed, setSeed] = useRecoilState(seedState);
    const [ckpt, setCkpt] = useRecoilState(ckptState);
    const [vae, setVae] = useRecoilState(vaeState);

    const [randomSeed, setRandomSeed] = useRecoilState(randomSeedState);

    const lora = useRecoilValue(loraState);
    const removeBackground = useRecoilValue(removeBackgroundState);
    const useRemovedBackground = useRecoilValue(useRemovedBackgroundState);
    const setSettings = useSetRecoilState(exifDataSelector);

    const goToModelFolder = useCallback(() => {
        if (modelDirs !== '') {
            window.api.showInExplorer(modelDirs);
        }
    }, [modelDirs]);

    const setSettingsWithToast = (results: CheckSettingsType) => {
        if(results[1].status !== 'error') {
            setSettings(results[0]);
        }
        toast(results[1]);
    }

    return (
        <Flex pr="10" width="450px">
            <Box m="-1" p={4} rounded="md">
                <VStack className="sd-settings" spacing={3}>
                    <HStack>
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

                        <FormControl width="200px" className="num-images-input">
                            <FormLabel htmlFor="n_iter">
                                № of Images
                            </FormLabel>

                            <NumberInput
                                id="n_iter"
                                name="n_iter"
                                min={1}
                                onChange={setIterations}
                                value={iterations}
                                variant="outline"
                            >
                                <NumberInputField id="n_iter" />
                            </NumberInput>
                        </FormControl>
                    </HStack>

                    <AspectRatio />

                    <FormControl className="model-ckpt-input">
                        <FormLabel htmlFor="Ckpt">
                            <HStack>
                                <Text>
                                    Model
                                </Text>
                                <Button
                                    fontSize='xs'
                                    leftIcon={<FaFolder/>}
                                    _hover={{ cursor: 'pointer' }}
                                    onClick={goToModelFolder}
                                    variant="outline"
                                    size="xs"
                                    >
                                    View
                                </Button>
                                {cloudMode
                                    ? <Icon as={IoMdCloud} />
                                    : null}
                            </HStack>
                        </FormLabel>

                        <Select
                            id="ckpt"
                            name="ckpt"
                            onChange={(event) => setCkpt(event.target.value)}
                            value={ckpt}
                            variant="outline"
                        >
                            <option value="">
                                Choose Your Model Weights
                            </option>

                            {models.ckpts.map((ckpt_option, i) => (<option
                                key={i}
                                value={ckpt_option}
                            >
                                {ckpt_option}
                            </option>))}
                        </Select>
                    </FormControl>

                    <SDSettingsAccordion header={`Advanced Settings`}>
                        { tab === 'paint' ? (
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
                                    max={0.95}
                                    min={0.03}
                                    name="strength"
                                    onChange={setStrength}        
                                    step={0.01}
                                    value={strength}
                                    variant="outline"
                                >
                                    <SliderTrack bg="#EEEEEE">
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
                        ) : null }

                        <HStack mt={4} alignItems="end">
                            <FormControl width="60%" className="steps-input">
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
                                    onChange={setSteps}
                                    value={steps}
                                    variant="outline"
                                >
                                    <NumberInputField id="steps" />
                                </NumberInput>
                            </FormControl>
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
                                    name="cfg_scale"
                                    onChange={setCfg}         
                                    value={cfg}
                                    variant="outline"
                                >
                                    <NumberInputField id="cfg_scale" />
                                </NumberInput>
                            </FormControl>
                        </HStack>
                        <HStack alignItems="end">
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
                                    <option value="ddim">
                                        DDIM
                                    </option>

                                    <option value="dpmpp_2m">
                                        DPM++ 2M Karras
                                    </option>

                                    <option value="dpmpp_2s_ancestral">
                                        DPM++ 2S Ancestral Karras
                                    </option>

                                    <option value="euler">
                                        Euler
                                    </option>

                                    <option value="euler_a">
                                        Euler Ancestral
                                    </option>

                                    <option value="dpmpp_sde">
                                        DPM++ SDE
                                    </option>

                                    <option value="dpm_2">
                                        DPM 2
                                    </option>

                                    <option value="dpm_a">
                                        DPM 2 Ancestral
                                    </option>

                                    <option value="lms">
                                        LMS
                                    </option>

                                    <option value="heun">
                                        Heun
                                    </option>

                                    <option value="plms">
                                        PLMS
                                    </option>
                                </Select>
                            </FormControl>
                            <FormControl width='55%' className="clip-skip-input">
                                <HStack>
                                    <FormLabel htmlFor="clip-skip">
                                        Clip Skip:
                                    </FormLabel>

                                    <Spacer />

                                    <Tooltip
                                        fontSize="md"
                                        label="Some anime models/loras prefer to have a clip skip of 2"
                                        placement="left"
                                        shouldWrapChildren
                                    >
                                        <FaQuestionCircle color="#777" />
                                    </Tooltip>
                                </HStack>

                                <NumberInput
                                    id="clip_skip"
                                    min={0}
                                    name="clip_skip"
                                    onChange={setClipSkip}         
                                    value={clipSkip}
                                    variant="outline"
                                >
                                    <NumberInputField id="clip_skip" />
                                </NumberInput>
                            </FormControl>     
                        </HStack>

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
                                value={vae}
                                variant="outline"
                            >
                                <option value="">
                                    No vae
                                </option>
                                {models.vaes.map((ckpt_option, i) => (<option
                                    key={i}
                                    value={ckpt_option}
                                >
                                    {ckpt_option}
                                </option>))}
                            </Select>
                        </FormControl>
                    </SDSettingsAccordion> 

                    <SDSettingsAccordion header={`Loras ${lora.length ? `(${lora.length})` : ``}`}>
                        <LoraSelector cloudMode={cloudMode} options={models.loras} />
                    </SDSettingsAccordion> 

                    <SDSettingsAccordion header={`ControlNet ${controlnet !== 'none' ? `(${controlnet})` : ``}`}>
                        <Controlnet />
                    </SDSettingsAccordion>   

                    <SDSettingsAccordion header={`Background Removal ${useRemovedBackground ? `(${removeBackground})` : ``}`}>
                        <RemoveBackground />
                    </SDSettingsAccordion>

                    <SDSettingsAccordion header='Highres settings'>
                        <HighresUpscale />
                    </SDSettingsAccordion>

                    <ButtonGroup isAttached variant="outline">
                        <Tooltip label="Upload">
                            <Button
                                onClick={() => {
                                    window.api.uploadSettings()
                                        .then(settings => checkSettings(settings, models))
                                        .then(setSettingsWithToast);
                                }}
                                aria-label="upload"
                                variant="outline"
                            >
                                Load SDSettings from file
                            </Button>
                        </Tooltip>
                        <Tooltip label="Paste from Clipboard">
                            <IconButton
                                aria-label="Paste from Clipboard"
                                variant="outline"
                                icon={<FaClipboardList />}
                                onClick={() => {
                                    navigator.clipboard.readText()
                                        .then(getExifData)
                                        .then(settings => checkSettings(settings, models))
                                        .then(setSettingsWithToast);
                                }}
                            />
                        </Tooltip>
                    </ButtonGroup>
                </VStack>
            </Box>
        </Flex>
    );
}
export default SDSettings;
