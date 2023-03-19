import React, { useCallback, useState, useEffect, useContext } from 'react';
import { useRecoilState, useRecoilValue } from 'recoil';
import * as atom from '../atoms/atoms';
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
    Accordion,
    AccordionButton,
    AccordionIcon,
    AccordionItem,
    AccordionPanel,
    Button
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';
import { IoMdCloud } from 'react-icons/io';
import { batchNameState, cfgState, ckptState, controlnetState, initImageState, iterationsState, loraState, modelsDirState, randomSeedState, samplerState, seedState, stepsState, strengthState, usePreprocessedControlnetState, vaeState } from '../SettingsManager';
import LoraSelector from './LoraSelector';
import { AspectRatio } from './SDSettings/AspectRatio';
import { SocketContext } from '../socket';
import ControlnetPreview from './ControlnetPreview/ControlnetPreview';

function SDSettings () {
    const socket = useContext(SocketContext);

    const cloudMode = useRecoilValue(atom.cloudModeState);

    const modelDirs = useRecoilValue(modelsDirState);
    const [ckpts, setCkpts] = useState([]);
    const [vaes, setVaes] = useState([]);
    const [loras, setLoras] = useState([]);

    const [batchName, setBatchName] = useRecoilState(batchNameState);
    const [iterations, setIterations] = useRecoilState(iterationsState);
    const [steps, setSteps] = useRecoilState(stepsState);
    const [cfg, setCfg] = useRecoilState(cfgState);
    const [sampler, setSampler] = useRecoilState(samplerState);
    const [controlnet, setControlnet] = useRecoilState(controlnetState);
    const [strength, setStrength] = useRecoilState(strengthState);
    const [seed, setSeed] = useRecoilState(seedState);
    const [ckpt, setCkpt] = useRecoilState(ckptState);
    const [vae, setVae] = useRecoilState(vaeState);

    const [randomSeed, setRandomSeed] = useRecoilState(randomSeedState);
    const [usePreprocessedControlnet, setUsePreprocessedControlnet] = useRecoilState(usePreprocessedControlnetState);

    const lora = useRecoilValue(loraState);
    const initImage = useRecoilValue(initImageState);
    const controlnetPreview = useRecoilValue(atom.controlnetPreviewState);

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
                                fontSize="sm"
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

                    <Accordion allowToggle border="none" bg="transparent" width="100%">
                        <AccordionItem border="none">
                            <AccordionButton p={0} bg="transparent" _hover={{ bg: 'transparent' }}>
                                <Box width="100%" flex="1" textAlign="start">
                                    <h2><b>Advanced Settings</b></h2>
                                </Box>
                                <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel width="100%" p={0} mt={4} mb={2} bg="transparent">
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
                                    <FormControl className="steps-input">
                                        <HStack>
                                            <FormLabel fontSize='sm' htmlFor="steps">
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
                                    <FormControl className="cfg-scale-input">
                                        <HStack>
                                            <FormLabel fontSize='sm' htmlFor="cfg_scale">
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
                            </AccordionPanel>
                        </AccordionItem>
                    </Accordion>

                    <Accordion allowToggle border="none" bg="transparent" width="100%">
                        <AccordionItem border="none">
                            <AccordionButton p={0} bg="transparent" _hover={{ bg: 'transparent' }}>
                                <Box width="100%" flex="1" textAlign="start">
                                    <h2><b>{`Loras ${lora.length ? `(${lora.length})` : ``}`}</b></h2>
                                </Box>
                                <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel p={0} mt={4} mb={2} width="100%" bg="transparent">
                                <LoraSelector cloudMode={cloudMode} options={loras} />
                            </AccordionPanel>
                        </AccordionItem>
                    </Accordion>
                    <Accordion allowToggle border="none" bg="transparent" width="100%">
                        <AccordionItem border="none">
                            <AccordionButton p={0} bg="transparent" _hover={{ bg: 'transparent' }}>
                                <Box width="100%" flex="1" textAlign="start">
                                    <h2><b>{`ControlNet ${controlnet !== 'none' ? `(${controlnet})` : ``}`}</b></h2>
                                </Box>
                                <AccordionIcon />
                            </AccordionButton>
                            <AccordionPanel p={0} mt={4} mb={2} width="100%" bg="transparent">
                                <VStack>
                                    <FormControl className="controlnet-input">
                                        <HStack>
                                            <FormLabel htmlFor="Controlnet">
                                                Choose your controlnet
                                            </FormLabel>
                                        </HStack>
                                        <HStack>
                                            <Select
                                                id="controlnet"
                                                name="controlnet"
                                                onChange={(event) => setControlnet(event.target.value)}
                                                value={controlnet}
                                                variant="outline"
                                            >
                                                <option value="none">
                                                    None
                                                </option>

                                                <option value="canny">
                                                    Canny
                                                </option>

                                                <option value="pose">
                                                    Pose
                                                </option>

                                                <option value="depth">
                                                    Depth
                                                </option>
                                                
                                                <option value="hed">
                                                    HED
                                                </option>

                                                <option value="normal">
                                                    Normal
                                                </option>

                                                <option value="scribble">
                                                    Scribble
                                                </option>
                                            </Select>
                                            <Button
                                                variant='outline'
                                                disabled={controlnet==='none' || usePreprocessedControlnet}
                                                onClick={()=>{
                                                    socket.emit('preview_controlnet', {initImage, controlnet})
                                                }
                                            }
                                        >
                                            Preview
                                            </Button>
                                        </HStack>
                                    </FormControl>
                                    <HStack>
                                    <FormLabel htmlFor="use_random_seed">
                                        Use Preprocessed ControlNet
                                    </FormLabel>
                                    <Checkbox
                                        id="use_preprocessed_controlnet"
                                        isChecked={usePreprocessedControlnet}
                                        name="use_preprocessed_controlnet"
                                        onChange={() => {
                                            setUsePreprocessedControlnet((usePreprocessedControlnet) => !usePreprocessedControlnet);
                                        }}
                                        pb="12px"
                                        />

                                    </HStack>
  
                                    {controlnetPreview.length > 0 && 
                                        <ControlnetPreview></ControlnetPreview> 
                                    }
                                </VStack>
                            </AccordionPanel>
                        </AccordionItem>
                    </Accordion>
                    
                </VStack>
            </Box>
        </Flex>
    );
}
export default SDSettings;
