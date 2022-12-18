import { React, useState, useEffect } from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import axios from 'axios';
import {
    createStandaloneToast,
    Box,
    Button,
    ButtonGroup,
    IconButton,
    Flex,
    FormControl,
    FormLabel,
    Input,
    VStack,
    HStack,
    Tooltip,
    NumberInputField,
    NumberInput,
    Text,
    Stack,
    Select,
    Slider,
    SliderThumb,
    SliderFilledTrack,
    SliderTrack
} from '@chakra-ui/react';
import {
    FaQuestionCircle,
    FaTrashAlt
} from 'react-icons/fa';
function Upscale () {
    const { ToastContainer, toast } = createStandaloneToast();
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    const [upscale_images, setUpscaleImages] = useState('');
    const [upscale_dest, setUpscaleDest] = useState('');
    const [upscaler, setUpscaler] = useState('ESRGAN');
    const [upscale_factor, setUpscaleFactor] = useState(2);
    const [upscale_strength, setUpscaleStrength] = useState(0.5);

    const chooseUploadPath = (event) => {
        window.chooseImages().then((result) => {
            setUpscaleImages(result);
        });
    };

    const chooseDestPath = (event) => {
        window.chooseUploadPath().then((result) => {
            setUpscaleDest(result);
        });
    };

    const upscale = (event) => {
        toast({
            title: 'Recieved!',
            description: 'You\'ll get a notification 🔔 when your upscale is ready! (First time upscales make take longer while it downloads the models)',
            status: 'success',
            position: 'top',
            duration: 3000,
            isClosable: false,
            containerStyle: {
                pointerEvents: 'none'
            }
        });
        const output = {
            upscale_images,
            upscaler,
            upscale_factor,
            upscale_strength,
            upscale_dest
        };
        axios.post(
            'http://127.0.0.1:5300/upscale',
            output,
            {
                headers: { 'Content-Type': 'application/json' }
            }
        ).then((result) => {
            if (result.data.status === 'Failure') {
                toast({
                    title: 'Upscale Failed',
                    description: result.data.status_message,
                    status: 'error',
                    position: 'top',
                    duration: 5000,
                    isClosable: false,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
            } else {
                toast({
                    title: 'Upscale Completed',
                    status: 'success',
                    position: 'top',
                    duration: 2000,
                    isClosable: false,
                    containerStyle: {
                        pointerEvents: 'none'
                    }
                });
            }
        });
    };


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
                        className="upscale"
                        spacing={5}>
                        <FormControl
                            className="upscale-images-input"
                            width="full">
                            <HStack>
                                <Tooltip
                                    fontSize="md"
                                    label="You can select all images in folder with Ctrl+A, it will ignore folders in upscaling"
                                    mt="3"
                                    placement="right"
                                    shouldWrapChildren>
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>

                                <FormLabel htmlFor="upscale_images">
                                    Choose images to upscale
                                </FormLabel>
                            </HStack>

                            <HStack>
                                <Input
                                    id="upscale_images"
                                    name="upscale_images"
                                    onChange={(event) => setUpscaleImages(event.target.value)}
                                    type="text"
                                    value={upscale_images}
                                    variant="outline"
                                />

                                <ButtonGroup
                                    isAttached
                                    pl="10px"
                                    variant="outline">
                                    <Button onClick={chooseUploadPath}>
                                        Choose
                                    </Button>

                                    <IconButton
                                        aria-label="Clear Init Image"
                                        icon={<FaTrashAlt />}
                                        onClick={(event) => setUpscaleImages('')} />
                                </ButtonGroup>
                            </HStack>
                        </FormControl>

                        <FormControl
                            className="upscale-dest-input"
                            width="full">
                            <FormLabel htmlFor="upscale_dest">
                                Choose upscale destination (leave blank to default to same directory as images)
                            </FormLabel>

                            <HStack>
                                <Input
                                    id="upscale_dest"
                                    name="upscale_dest"
                                    onChange={(event) => setUpscaleDest(event.target.value)}
                                    type="text"
                                    value={upscale_dest}
                                    variant="outline"
                                />

                                <ButtonGroup
                                    isAttached
                                    pl="10px"
                                    variant="outline">
                                    <Button onClick={chooseDestPath}>
                                        Choose
                                    </Button>

                                    <IconButton
                                        aria-label="Clear Init Image"
                                        icon={<FaTrashAlt />}
                                        onClick={(event) => setUpscaleDest('')} />
                                </ButtonGroup>
                            </HStack>
                        </FormControl>

                        <HStack >
                            <Tooltip
                                fontSize="md"
                                label={<Stack>
                                    <Text >
                                        GFPGAN fixes faces while upscaling
                                        {' '}
                                    </Text>

                                    <Text >
                                        RealESRGAN is a general upscaler
                                    </Text>

                                    <Text >
                                        RealESRGAN-anime is a better for anime
                                    </Text>
                                </Stack>}
                                mt="3"
                                placement="right"
                                shouldWrapChildren>
                                <FaQuestionCircle color="#777" />
                            </Tooltip>

                            <FormLabel htmlFor="upscaler">
                                Choose Upscaler
                            </FormLabel>
                        </HStack>

                        <HStack>
                            <FormControl className="upscaler-input">
                                <Select
                                    id="upscaler"
                                    name="upscaler"
                                    onChange={(event) => setUpscaler(event.target.value)}
                                    value={upscaler}
                                    variant="outline"
                                    w="300px"
                                >
                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="Choose Upscaler">
                                        Choose Upscaler
                                    </option>

                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="GFPGANv1.3">
                                        GFPGANv1.3
                                    </option>

                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="GFPGANv1.4">
                                        GFPGANv1.4
                                    </option>

                                    {/* <option value='CodeFormer'>CodeFormer</option> */}

                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="RestoreFormer">
                                        RestoreFormer
                                    </option>

                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="RealESRGAN">
                                        RealESRGAN
                                    </option>

                                    <option
                                        style={{ backgroundColor: '#080B16' }}
                                        value="RealESRGAN-Anime">
                                        RealESRGAN-Anime
                                    </option>
                                </Select>
                            </FormControl>

                            {/* {upscaler === 'CodeFormer' ?
                  <FormControl>
                    <HStack>
                    <Tooltip shouldWrapChildren  placement='top' label='CodeFormer Upscale Strength' fontSize='md'>
                        <FaQuestionCircle color='#777'/>
                        </Tooltip>
                        <FormLabel  htmlFor='upscale_strength'>Upscale Strength:</FormLabel>
                    </HStack>
                    <Slider
                        id='upscale_strength'
                        name='upscale_strength'
                        variant='outline'
                        defaultValue={0.5} min={0.01} max={1.0} step={0.01} colorScheme='teal'
                        onChange={(v) => {setUpscaleStrength(v)}}
                        >
                        <SliderTrack bg='red.100'>
                          <Box position='relative' right={10} />
                            <SliderFilledTrack bg='tomato' />
                          </SliderTrack>
                        <Tooltip
                          bg='teal.500'
                          color='white'
                          placement='right'
                          isOpen={true}
                          label={`${upscale_strength}`}
                          >
                        <SliderThumb />
                        </Tooltip>
                    </Slider>
                    </FormControl>
                      : <></>
                    } */}
                        </HStack>

                        <FormControl className="upscale-factor-input">
                            <HStack>
                                <Tooltip
                                    fontSize="md"
                                    label={<Stack>
                                        <Text >
                                            Multiplies your Width and Height
                                        </Text>

                                        <Text >
                                            (ex: a multiplier of 2 will make 512x512 into 1024x1024)
                                        </Text>
                                    </Stack>}
                                    placement="top"
                                    shouldWrapChildren>
                                    <FaQuestionCircle color="#777" />
                                </Tooltip>

                                <FormLabel htmlFor="upscale_factor">
                                    Upscale Multiplier (whole numbers only)
                                </FormLabel>
                            </HStack>

                            <NumberInput
                                id="upscale_factor"
                                min={1}
                                name="upscale_factor"
                                onChange={(v) => {
                                    setUpscaleFactor(v);
                                }}
                                step={1}
                                value={upscale_factor}
                                variant="outline"
                                w="200px"
                            >
                                <NumberInputField id="upscale_factor" />
                            </NumberInput>
                        </FormControl>


                        <Button
                            className="upscale-button"
                            onClick={upscale}>
                            Upscale
                        </Button>
                    </VStack>
                </form>
            </Box>
        </Flex>
    );
}

export default Upscale;
