import { useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Box,
    Button,
    Flex,
    VStack,
    Text,
    createStandaloneToast,
    FormControl,
    FormLabel,
    HStack,
    Select,
    Radio,
    RadioGroup,
    Stack,
    Input,
    Checkbox,
    Slider,
    SliderMark,
    SliderTrack,
    SliderThumb,
    SliderFilledTrack,
    RangeSlider,
    RangeSliderMark,
    RangeSliderTrack,
    RangeSliderFilledTrack,
    RangeSliderThumb,
    Tooltip
} from '@chakra-ui/react';
import { FaQuestionCircle } from 'react-icons/fa';

export const ModelMerger = () => {
    const { ToastContainer, toast } = createStandaloneToast();
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

    const [progress, setProgress] = useState(-1);
    const [stage, setStage] = useState('');
    const [ckpts, setCkpts] = useState([]);
    const [ckpt_dir, setCkptDir] = useRecoilState(atom.ckptDirState);

    const [modelA, setModelA] = useState('');
    const [modelB, setModelB] = useState('');
    const [modelC, setModelC] = useState('');
    const [interpolation, setInterpolation] = useState('weighted_sum');
    const [alpha, setAlpha] = useState(0);
    const [alphaRange, setAlphaRange] = useState([33, 66]);

    const [fullrange, setFullrange] = useState(false);
    const [filename, setFilename] = useState('');

    const labelStyles = {
        mt: '2',
        ml: '-2.5',
        fontSize: 'sm'
    };

    const submitMain = () => {
        const data = JSON.stringify({
            modelA,
            modelB,
            modelC: interpolation === 'add_difference' ? modelC : '',
            method: interpolation,
            fullrange,
            alpha,
            filename,
            steps: fullrange ? alpha : 0
        });
        console.log(data);
        window.mergeModels(data).then((res) => {
            if (res === 0) {
                toast({
                    title: 'Merged successfully',
                    status: 'success',
                    position: 'top',
                    duration: 1500,
                    isClosable: false,
                    containerStyle: { pointerEvents: 'none' }
                });
            } else {
                toast({
                    title: 'Failed',
                    status: 'error',
                    position: 'top',
                    duration: 1500,
                    isClosable: false,
                    containerStyle: { pointerEvents: 'none' }
                });
            }
        }).catch((err) => {
            toast({
                title: `There was an error: ${err}`,
                status: 'error',
                position: 'top',
                duration: 1500,
                isClosable: false,
                containerStyle: { pointerEvents: 'none' }
            });
        });
    };

    const getCkpts = () => {
        window.getCkpts(ckpt_dir).then((result) => {
            // console.log(result);
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

    return (
        <Flex
            align="center"
            justify="center"
            ml={navSize === 'large'
                ? '130px'
                : '0px'}
            transition="all .25s ease"
            width="100%">

            <Box
                height="90%"
                ml="30px"
                p={4}
                rounded="md"
                width="75%">

                <form>
                    <VStack
                        align="flex-start"
                        spacing={5}>

                        <FormControl
                            width="full">
                            <HStack>
                                <Tooltip
                                    fontSize="md"
                                    label="These determine how you would like to handle the model merges. Note that you can use 3 models if you select Add Difference."
                                    mt="3"
                                    placement="right"
                                    shouldWrapChildren>
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>

                                <FormLabel htmlFor="upscale_images">
                                    Select interpolation method
                                </FormLabel>
                            </HStack>

                            <HStack>
                                <RadioGroup
                                    onChange={setInterpolation}
                                    value={interpolation}>
                                    <Stack direction="row">
                                        <Radio value="weighted_sum">
                                            Weighted Sum
                                        </Radio>

                                        <Radio value="sigmoid">
                                            Sigmoid
                                        </Radio>

                                        <Radio value="inverse_sigmoid">
                                            Inverse Sigmoid
                                        </Radio>

                                        <Radio value="add_difference">
                                            Add Difference
                                        </Radio>
                                    </Stack>
                                </RadioGroup>
                            </HStack>
                        </FormControl>

                        <FormControl width="full">
                            <FormLabel htmlFor="Ckpt">
                                <HStack>
                                    <Text>
                                        Model 1
                                    </Text>
                                </HStack>
                            </FormLabel>

                            <Select
                                id="ckpt"
                                name="ckpt"
                                onChange={(event) => setModelA(event.target.value)}
                                onMouseEnter={getCkpts}
                                value={modelA}
                                variant="outline"
                            >
                                {ckpts.length > 0
                                    ? <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value=""
                                    >
                                        Select model weights
                                    </option>
                                    : <></>}

                                {ckpts?.map((ckpt_option, i) => (<option
                                    key={i}
                                    style={{ 'backgroundColor': '#080B16' }}
                                    value={ckpt_option}
                                >
                                    {ckpt_option}
                                </option>))}
                            </Select>
                        </FormControl>

                        <FormControl width="full">
                            <FormLabel htmlFor="Ckpt">
                                <HStack>
                                    <Text>
                                        Model 2
                                    </Text>
                                </HStack>
                            </FormLabel>

                            <Select
                                id="ckpt"
                                name="ckpt"
                                onChange={(event) => setModelB(event.target.value)}
                                onMouseEnter={getCkpts}
                                value={modelB}
                                variant="outline"
                            >
                                {ckpts.length > 0
                                    ? <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value=""
                                    >
                                        Select model weights
                                    </option>
                                    : <></>}

                                {ckpts?.map((ckpt_option, i) => (<option
                                    key={i}
                                    style={{ 'backgroundColor': '#080B16' }}
                                    value={ckpt_option}
                                >
                                    {ckpt_option}
                                </option>))}
                            </Select>
                        </FormControl>

                        {interpolation === 'add_difference' && <FormControl
                            width="full">
                            <FormLabel htmlFor="Ckpt">
                                <HStack>
                                    <Text>
                                        Model 3
                                    </Text>
                                </HStack>
                            </FormLabel>

                            <Select
                                id="ckpt"
                                name="ckpt"
                                onChange={(event) => setModelC(event.target.value)}
                                onMouseEnter={getCkpts}
                                value={modelC}
                                variant="outline"
                            >
                                {ckpts.length > 0
                                    ? <option
                                        style={{ 'backgroundColor': '#080B16' }}
                                        value=""
                                    >
                                        Select model weights
                                    </option>
                                    : <></>}

                                {ckpts?.map((ckpt_option, i) => (<option
                                    key={i}
                                    style={{ 'backgroundColor': '#080B16' }}
                                    value={ckpt_option}
                                >
                                    {ckpt_option}
                                </option>))}
                            </Select>
                        </FormControl>}

                        <FormControl>
                            <HStack>
                                <Checkbox
                                    onChange={() => {
                                        setFullrange(!fullrange);
                                    }}
                                    value={fullrange}>
                                    Use full alpha range
                                </Checkbox>

                                <Tooltip
                                    fontSize="md"
                                    label="Merge the model for every % value from 5 to 95, every 5 steps"
                                    mt="3"
                                    placement="right"
                                    shouldWrapChildren>
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>
                            </HStack>
                        </FormControl>

                        {!fullrange && interpolation === 'add_difference'
                            ? <FormControl
                                pt={5}
                                width="full">
                                <RangeSlider
                                    aria-label="slider-ex-6"
                                    defaultValue={[33, 66]}
                                    max={100}
                                    min={0}
                                    onChange={(val) => setAlphaRange(val)}
                                    step={1}
                                    value={alphaRange}
                                >
                                    <RangeSliderMark
                                        value={25}
                                        {...labelStyles}>
                                        25%
                                    </RangeSliderMark>

                                    <RangeSliderMark
                                        value={50}
                                        {...labelStyles}>
                                        50%
                                    </RangeSliderMark>

                                    <RangeSliderMark
                                        value={75}
                                        {...labelStyles}>
                                        75%
                                    </RangeSliderMark>

                                    <RangeSliderMark
                                        bg="blue.500"
                                        color="white"
                                        ml="-5"
                                        mt="-10"
                                        textAlign="center"
                                        value={alphaRange[0]}
                                        w="12"
                                    >
                                        {alphaRange[0]}
                                        %
                                    </RangeSliderMark>

                                    <RangeSliderMark
                                        bg="blue.500"
                                        color="white"
                                        ml="-5"
                                        mt="-10"
                                        textAlign="center"
                                        value={alphaRange[1]}
                                        w="12"
                                    >
                                        {alphaRange[1]}
                                        %
                                    </RangeSliderMark>


                                    <RangeSliderTrack>
                                        <RangeSliderFilledTrack />
                                    </RangeSliderTrack>

                                    <RangeSliderThumb index={0} />

                                    <RangeSliderThumb index={1} />
                                </RangeSlider>
                            </FormControl>
                            : !fullrange
                                ? <FormControl
                                    pt={5}
                                    width="full">
                                    <Slider
                                        aria-label="slider-ex-6"
                                        // maxLabel={modelB}
                                        // minLabel={modelA}
                                        onChange={(val) => setAlpha(val)}
                                        step={1}
                                        value={alpha}
                                    >
                                        <SliderMark
                                            value={25}
                                            {...labelStyles}>
                                            25%
                                        </SliderMark>

                                        <SliderMark
                                            value={50}
                                            {...labelStyles}>
                                            50%
                                        </SliderMark>

                                        <SliderMark
                                            value={75}
                                            {...labelStyles}>
                                            75%
                                        </SliderMark>

                                        <SliderMark
                                            bg="blue.500"
                                            color="white"
                                            ml="-5"
                                            mt="-10"
                                            textAlign="center"
                                            value={alpha}
                                            w="12"
                                        >
                                            {alpha}
                                            %
                                        </SliderMark>

                                        <SliderTrack>
                                            <SliderFilledTrack />
                                        </SliderTrack>

                                        <SliderThumb />
                                    </Slider>
                                </FormControl>
                                : <></>}

                        <FormControl width="full">
                            <Input
                                onChange={(e) => setFilename(e.target.value)}
                                placeholder="Output file name"
                                size="md"
                                value={filename} />
                        </FormControl>

                        <FormControl
                            align="center"
                            width="full" >

                            <VStack spacing={3}>
                                <Button
                                    className="run-button"
                                    ml={2}
                                    onClick={submitMain}
                                    width="200px">
                                    Merge
                                </Button>
                            </VStack>
                        </FormControl>
                    </VStack>
                </form>

            </Box>
        </Flex>
    );
};
