import React, { useCallback, useState, useEffect } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
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
    Icon
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import { IoMdCloud } from 'react-icons/io';
import {
    batchNameState,
    cfgState,
    ckptState,
    clipSkipState,
    controlnetState,
    iterationsState,
    loraState,
    modelsDirState,
    randomSeedState,
    removeBackgroundState,
    samplerState,
    seedState,
    stepsState,
    strengthState,
    useRemovedBackgroundState,
    vaeState
} from '../../SettingsManager';
import LoraSelector from './Lora/LoraSelector';
import { AspectRatio } from './AspectRatio';
import Controlnet from './Controlnet/Controlnet';
import RemoveBackground from './RemoveBackground/RemoveBackground';
import { SDSettingsAccordion } from './SDSettingsAccordion';

function SDSettings () {
    const cloudMode = useRecoilValue(atom.cloudModeState);

    const modelDirs = useRecoilValue(modelsDirState);
    const [ckpts, setCkpts] = useState([]);
    const [vaes, setVaes] = useState([]);
    const [loras, setLoras] = useState([]);

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

    const getCkpts = useCallback(() => {
        window.api.getCkpts(modelDirs).then(setCkpts);
        window.api.getVaes(modelDirs).then(setVaes);
        window.api.getLoras(modelDirs).then(setLoras);
        
    }, [modelDirs]);

    useEffect(getCkpts, [getCkpts]);

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
                                ? <option value="">
                                    Choose Your Model Weights
                                </option>
                                : <></>}

                            {ckpts.map((ckpt_option, i) => (<option
                                key={i}
                                value={ckpt_option}
                            >
                                {ckpt_option}
                            </option>))}
                        </Select>
                    </FormControl>

                    <SDSettingsAccordion header={`Advanced Settings`}>
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
                                        Prompt Strength:
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
                                onMouseEnter={getCkpts}
                                value={vae}
                                variant="outline"
                            >
                                <option value="">
                                    No vae
                                </option>
                                {vaes.map((ckpt_option, i) => (<option
                                    key={i}
                                    value={ckpt_option}
                                >
                                    {ckpt_option}
                                </option>))}
                            </Select>
                        </FormControl>
                    </SDSettingsAccordion> 

                    <SDSettingsAccordion header={`Loras ${lora.length ? `(${lora.length})` : ``}`}>
                        <LoraSelector cloudMode={cloudMode} options={loras} />
                    </SDSettingsAccordion> 

                    <SDSettingsAccordion header={`ControlNet ${controlnet !== 'none' ? `(${controlnet})` : ``}`}>
                        <Controlnet />
                    </SDSettingsAccordion>   

                    <SDSettingsAccordion header={`Background Removal ${useRemovedBackground ? `(${removeBackground})` : ``}`}>
                        <RemoveBackground />
                    </SDSettingsAccordion>             
                </VStack>
            </Box>
        </Flex>
    );
}
export default SDSettings;
