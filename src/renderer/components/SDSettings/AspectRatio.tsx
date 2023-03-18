import React, { useEffect } from "react";
import {
    VStack,
    Icon,
    Select,
    Input,
    useToast,
    Box,
    FormControl,
    FormLabel,
    HStack,
    Slider,
    SliderFilledTrack,
    SliderThumb,
    SliderTrack,
    Tooltip
} from "@chakra-ui/react";
import { FiLink } from "react-icons/fi";
import { useRecoilState, useRecoilValue } from "recoil";
import { aspectRatioSelectionState } from "../../atoms/atoms";
import { aspectRatioState, heightState, initImageState, widthState } from "../../SettingsManager";

export const AspectRatio = () => {
    const toast = useToast({});

    const [width, setWidth] = useRecoilState(widthState);
    const [height, setHeight] = useRecoilState(heightState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(aspectRatioSelectionState);

    const [aspectRatio, setAspectRatio] = useRecoilState(aspectRatioState);
    const initImage = useRecoilValue(initImageState);

    useEffect(() => {
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
    }, [width, aspectRatio]);

    return (
        <Box className="size-input" width="100%">
            <HStack>
                <VStack width="100%">
                    <FormControl className="width-input">
                        <FormLabel justifyContent="center" htmlFor="Width">
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
                                placement="left"
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
                                placement="left"
                            >
                                <SliderThumb />
                            </Tooltip>
                        </Slider>
                    </FormControl>
                </VStack>

                <FormControl width="50%" className="aspect-ratio-input" marginBottom={2}>
                    <VStack>
                        <Icon as={FiLink}/>
                        <Select
                        id="aspect_ratio_selection"
                        name="aspect_ratio_selection"
                        onChange={(event) => {
                            setAspectRatioSelection(event.target.value);
                            if (event.target.value === "Init Image" && !initImage) {
                                //Switch to aspect ratio to none and print warning that no init image is set
                                setAspectRatioSelection("None");
                                setAspectRatio("None");
                                toast({
                                    title: "Invalid Aspect Ratio Selection",
                                    description: "Must upload Starting Image first to use its resolution",
                                    status: "error",
                                    position: "top",
                                    duration: 3000,
                                    isClosable: true,
                                    containerStyle: {
                                    pointerEvents: "none",
                                    },
                                });
                            } else if (event.target.value !== "Custom") {
                                setAspectRatio(event.target.value);
                            }
                        }}
                        value={aspectRatioSelection}
                        variant="outline"
                        >
                        <option style={{ backgroundColor: "#080B16" }} value="None">
                            None
                        </option>

                        {initImage.length && (
                            <option style={{ backgroundColor: "#080B16" }} value="Init Image">
                            Init Image
                            </option>
                        )}

                        <option style={{ backgroundColor: "#080B16" }} value="1:1">
                            1:1
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="1:2">
                            1:2
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="2:1">
                            2:1
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="4:3">
                            4:3
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="3:4">
                            3:4
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="16:9">
                            16:9
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="9:16">
                            9:16
                        </option>

                        <option style={{ backgroundColor: "#080B16" }} value="Custom">
                            Custom
                        </option>
                        </Select>

                        {aspectRatioSelection === "Custom" ? (
                        <Input
                            id="aspect_ratio"
                            name="aspect_ratio"
                            onChange={(event) => setAspectRatio(event.target.value)}
                            value={aspectRatio}
                            variant="outline"
                        />
                        ) : (
                        <></>
                        )}
                    </VStack>
                </FormControl>
            </HStack>
        </Box>
    )
}
