import React, { useCallback, useEffect } from "react";
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
import { getImageDimensions } from "../Utils/image";

const MAX_VALUE = 2048;
const MIN_VALUE = 256;
const STEP = 64;

const ASPECT_RATIOS = [
    '1:1',
    '1:2',
    '2:1',
    '4:3',
    '3:4',
    '16:9',
    '9:16',
    'Custom'
];

const getRatio = (aspectRatio: string, invert: boolean) => {
    try {
        const values = aspectRatio.split(':');
        const widthRatio = parseFloat(values[0]);
        const heightRatio = parseFloat(values[1]);

        if (isNaN(widthRatio) || isNaN(heightRatio)) {
            return 1;
        }

        return invert ? widthRatio / heightRatio : heightRatio / widthRatio;
    } catch {
        return 1;
    }
}

const getValue = (value: number, aspectRatio: string, invert: boolean) => {
    let newValue = Math.floor(value * getRatio(aspectRatio, invert) / STEP) * STEP;
    newValue = Math.min(MAX_VALUE, newValue);
    newValue = Math.max(MIN_VALUE, newValue);

    return newValue;
}

export const AspectRatio = () => {
    const toast = useToast({});

    const [width, setWidth] = useRecoilState(widthState);
    const [height, setHeight] = useRecoilState(heightState);
    const [aspectRatioSelection, setAspectRatioSelection] = useRecoilState(aspectRatioSelectionState);

    const [aspectRatio, setAspectRatio] = useRecoilState(aspectRatioState);
    const initImage = useRecoilValue(initImageState);

    const handleWidthChange = useCallback((value: number) => {
        if (value > 0) {
            if (aspectRatioSelection !== 'None') {
                setHeight(getValue(value, aspectRatio, false));
            }
            setWidth(value);
        }
    }, [aspectRatio, aspectRatioSelection]);

    const handleHeightChange = useCallback((value: number) => {
        if (value > 0) {
            if (aspectRatioSelection !== 'None') {
                setWidth(getValue(value, aspectRatio, true));
            }
            setHeight(value);
        }
    }, [aspectRatio, aspectRatioSelection]);

    useEffect(() => {
        if (aspectRatioSelection !== 'None') {
            setHeight(getValue(width, aspectRatio, false));
        }
    }, [aspectRatio, aspectRatioSelection])

    useEffect(() => {
        if(aspectRatioSelection === 'Init Image') {
            getImageDimensions(initImage).then(({ width, height }) => {
                setAspectRatio(`${width}:${height}`);
            });
        }
    }, [aspectRatioSelection, initImage]);

    return (
        <Box className="size-input" width="100%">
            <HStack>
                <VStack width="100%">
                    <FormControl className="width-input">
                        <FormLabel htmlFor="width">Width:</FormLabel>

                        <Slider
                            defaultValue={512}
                            id="width"
                            max={MAX_VALUE}
                            min={MIN_VALUE}
                            name="width"
                            onChange={handleWidthChange}                                
                            step={STEP}
                            value={width}
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
                                label={`${width}`}
                                placement="left"
                            >
                                <SliderThumb />
                            </Tooltip>
                        </Slider>
                    </FormControl>

                    <FormControl className="height-input">
                        <FormLabel htmlFor="Height">Height:</FormLabel>

                        <Slider
                            defaultValue={512}
                            max={MAX_VALUE}
                            min={MIN_VALUE}
                            onChange={handleHeightChange}                                        
                            step={STEP}
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
                                isOpen={true}
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
                                const aspectRatio = event.target.value;

                                setAspectRatioSelection(aspectRatio);
                                if (aspectRatio === "Init Image" && !initImage) {
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
                                } else if (aspectRatio !== "Custom") {
                                    setAspectRatio(aspectRatio);
                                }
                            }}
                            value={aspectRatioSelection}
                            variant="outline"
                        >
                            <option value="None">
                                None
                            </option>

                            {initImage.length && (
                                <option value="Init Image">
                                    Init Image
                                </option>
                            )}

                            { ASPECT_RATIOS.map(ratio => <option value={ratio} key={ratio}>{ratio}</option>) }
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
