import React, { useState, useCallback, useContext } from 'react';
import {
    Box,
    Button,
    ButtonGroup,
    IconButton,
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
    useToast,
    Flex,
    Icon
} from '@chakra-ui/react';
import {
    FaQuestionCircle,
    FaTrashAlt
} from 'react-icons/fa';
import { SocketContext } from '../socket';
import { useDropzone } from 'react-dropzone';
import { useRecoilValue } from 'recoil';
import { imageSavePathState, modelsDirState } from '../SettingsManager';

function Upscale () {
    const toast = useToast({});
    const [upscale_images, setUpscaleImages] = useState<string[]>([]);
    const [upscale_dest, setUpscaleDest] = useState('');
    const [upscaler, setUpscaler] = useState('');
    const [upscale_factor, setUpscaleFactor] = useState(2);
    const [upscale_strength, setUpscaleStrength] = useState(0.5);
    const imageSavePath = useRecoilValue(imageSavePathState);
    const modelsDir = useRecoilValue(modelsDirState);

    const socket = useContext(SocketContext);
    // Listen for a response from the server for the 'upscale' event
    socket.on('upscale_completed', function(response) {
        // Handle the response from the server here
        window.api.showItemInFolder(response.upscale_dest)
    });

    const chooseUploadPath = () => {
        window.api.chooseImages().then(setUpscaleImages);
    };

    const chooseDestPath = () => {
        window.api.chooseUploadPath().then(setUpscaleDest);
    };

    const Dropzone = () => {
        const toast = useToast();
        const onDrop = useCallback((acceptedFiles: File[]) => {
            setUpscaleImages(acceptedFiles.map(file => file.path))
            
            toast({
                title: "Files added!",
                status: "success",
                duration: 5000,
                isClosable: true,
            });
        }, [toast]);
      
        const { getRootProps, getInputProps } = useDropzone({ onDrop, useFsAccessApi: false });
      
        return (
          <Box
            borderWidth="1px"
            borderRadius="md"
            p="5"
            my="5"
            backgroundColor="#080B16"
            width='100%'
            {...getRootProps()}
          >
            <input multiple {...getInputProps()} />
            <Flex align="center" justify="center" height="150px">
              <Icon name="cloud-upload"/>
              <Text ml="2">Drop your files here or click to browse</Text>
            </Flex>
          </Box>
        );
      };

    const upscale = useCallback(() => {
        if (!upscaler.length){
            toast({
                title: 'Please select an upscaler',
                status: 'error',
                position: 'top',
                duration: 3000,
                isClosable: true,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });   
            return;
        }
        toast({
            title: 'Received!',
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
            upscale_dest,
            image_save_path: imageSavePath,
            models_dir: modelsDir
        };

        socket.emit('upscale', output);
    }, [socket, toast, upscale_dest, upscale_factor, upscale_images, upscale_strength, upscaler, imageSavePath, modelsDir]);
    
    return (
        <Box
            height="90%"
            ml="30px"
            p={4}
            rounded="md"
            width="100%">
            <form>
                <VStack
                    align="flex-start"
                    className="upscale"
                    spacing={5}>
                    <Dropzone />
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
                                onChange={(event) => setUpscaleImages(event.target.value.split(',')?.filter(e => e !== ''))}
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
                                    aria-label="Clear Upscale Images"
                                    icon={<FaTrashAlt />}
                                    onClick={() => setUpscaleImages([])} />
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
                                    aria-label="Clear Upscale Destination"
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
                                <option value="">
                                    Choose Upscaler
                                </option>

                                <option value="UltraSharp">
                                    4x UltraSharp
                                </option>

                                <option value="GFPGANv1.3">
                                    GFPGANv1.3
                                </option>

                                <option value="GFPGANv1.4">
                                    GFPGANv1.4
                                </option>

                                {/* <option value='CodeFormer'>CodeFormer</option> */}

                                <option value="RestoreFormer">
                                    RestoreFormer
                                </option>

                                <option value="RealESRGAN">
                                    RealESRGAN
                                </option>

                                <option value="RealESRGAN-Anime">
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
                            onChange={(v, n) => {
                                setUpscaleFactor(n);
                            }}
                            step={0.01}
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
    );
}

export default Upscale;
