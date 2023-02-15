import React, { FocusEvent, useEffect, useMemo, useState } from 'react';
import {
	FormControl,
	FormControlProps,
	FormLabel,
	FormLabelProps,
	HStack,
	NumberDecrementStepper,
	NumberIncrementStepper,
	NumberInput,
	NumberInputField,
	NumberInputFieldProps,
	NumberInputProps,
	NumberInputStepper,
	NumberInputStepperProps,
	Slider as ChakraSlider,
	SliderFilledTrack,
	SliderMark,
	SliderMarkProps,
	SliderThumb,
	SliderThumbProps,
	SliderTrack,
	SliderTrackProps,
	Tooltip,
	TooltipProps,
	useColorModeValue,
} from '@chakra-ui/react';
import { BiReset } from 'react-icons/bi';
import _ from 'lodash';
import { IconButton, IIconButtonProps } from '../IconButton/IconButton';

export type IFullSliderProps = {
	label: string;
	value: number;
	min?: number;
	max?: number;
	step?: number;
	onChange: (v: number) => void;
	withSliderMarks?: boolean;
	sliderMarkLeftOffset?: number;
	sliderMarkRightOffset?: number;
	withInput?: boolean;
	isInteger?: boolean;
	width?: string | number;
	inputWidth?: string | number;
	inputReadOnly?: boolean;
	withReset?: boolean;
	handleReset?: () => void;
	isResetDisabled?: boolean;
	isSliderDisabled?: boolean;
	isInputDisabled?: boolean;
	tooltipSuffix?: string;
	hideTooltip?: boolean;
	sliderFormControlProps?: FormControlProps;
	sliderFormLabelProps?: FormLabelProps;
	sliderMarkProps?: Omit<SliderMarkProps, 'value'>;
	sliderTrackProps?: SliderTrackProps;
	sliderThumbProps?: SliderThumbProps;
	sliderNumberInputProps?: NumberInputProps;
	sliderNumberInputFieldProps?: NumberInputFieldProps;
	sliderNumberInputStepperProps?: NumberInputStepperProps;
	sliderTooltipProps?: Omit<TooltipProps, 'children'>;
	sliderIconButtonProps?: IIconButtonProps;
};

export const Slider: React.FC<IFullSliderProps> = ({
	label,
	value,
	min = 1,
	max = 100,
	step = 1,
	onChange,
	width = '100%',
	tooltipSuffix = '',
	withSliderMarks = false,
	sliderMarkLeftOffset = 0,
	sliderMarkRightOffset = -7,
	withInput = false,
	isInteger = false,
	inputWidth = '5rem',
	inputReadOnly = true,
	withReset = false,
	hideTooltip = false,
	handleReset,
	isResetDisabled,
	isSliderDisabled,
	isInputDisabled,
	sliderFormControlProps,
	sliderFormLabelProps,
	sliderMarkProps,
	sliderTrackProps,
	sliderThumbProps,
	sliderNumberInputProps,
	sliderNumberInputFieldProps,
	sliderNumberInputStepperProps,
	sliderTooltipProps,
	sliderIconButtonProps,
	...rest
}) => {
	const [showTooltip, setShowTooltip] = useState(false);

	const [localInputValue, setLocalInputValue] = useState<string>(
		String(value),
	);

	const numberInputMax = useMemo(
		() => (sliderNumberInputProps?.max ? sliderNumberInputProps.max : max),
		[max, sliderNumberInputProps?.max],
	);

	useEffect(() => {
		if (String(value) !== localInputValue && localInputValue !== '') {
			setLocalInputValue(String(value));
		}
	}, [value, localInputValue, setLocalInputValue]);

	const handleInputBlur = (e: FocusEvent<HTMLInputElement>) => {
		const clamped = _.clamp(
			isInteger
				? Math.floor(Number(e.target.value))
				: Number(e.target.value),
			min,
			numberInputMax,
		);
		setLocalInputValue(String(clamped));
		onChange(clamped);
	};

	const handleInputChange = (v: number | string) => {
		setLocalInputValue(String(v));
		onChange(Number(v));
	};

	const handleResetDisable = () => {
		if (!handleReset) return;
		handleReset();
	};

	return (
		<FormControl
			display="flex"
			gap="1rem"
			alignItems="center"
			data-markers={withSliderMarks}
			{...sliderFormControlProps}>
			<FormLabel
				margin={0}
				fontWeight="bold"
				fontSize="0.9rem"
				minWidth="max-content"
				color={useColorModeValue('#282828', '#a0a2bc')}
				{...sliderFormLabelProps}>
				{label}
			</FormLabel>

			<HStack w="100%" gap={2}>
				<ChakraSlider
					aria-label={label}
					value={value}
					min={min}
					max={max}
					step={step}
					onChange={handleInputChange}
					onMouseEnter={() => setShowTooltip(true)}
					onMouseLeave={() => setShowTooltip(false)}
					focusThumbOnChange={false}
					isDisabled={isSliderDisabled}
					width={width}
					{...rest}>
					{withSliderMarks && (
						<>
							<SliderMark
								value={min}
								fontSize="0.75rem"
								fontWeight="bold"
								mt="0.3rem"
								ml={sliderMarkLeftOffset}
								{...sliderMarkProps}>
								{min}
							</SliderMark>
							<SliderMark
								value={max}
								fontSize="0.75rem"
								fontWeight="bold"
								mt="0.3rem"
								ml={sliderMarkRightOffset}
								{...sliderMarkProps}>
								{max}
							</SliderMark>
						</>
					)}

					<SliderTrack bg="#EEEEEE" {...sliderTrackProps}>
						<SliderFilledTrack bg="buttonPrimary" />
					</SliderTrack>

					<Tooltip
						hasArrow
						placement="top"
						isOpen={showTooltip}
						label={`${value}${tooltipSuffix}`}
						hidden={hideTooltip}
						{...sliderTooltipProps}>
						<SliderThumb {...sliderThumbProps} />
					</Tooltip>
				</ChakraSlider>

				{withInput && (
					<NumberInput
						min={min}
						max={numberInputMax}
						step={step}
						value={localInputValue}
						onChange={handleInputChange}
						onBlur={handleInputBlur}
						isDisabled={isInputDisabled}
						{...sliderNumberInputProps}>
						<NumberInputField
							// border="none"
							fontSize="0.9rem"
							fontWeight="bold"
							height="2rem"
							bg={useColorModeValue('#d0d2d4', '#101016')}
							border="2px solid"
							borderColor={useColorModeValue(
								'#c8c8c8',
								'#1e1e2e',
							)}
							width={inputWidth}
							readOnly={inputReadOnly}
							{...sliderNumberInputFieldProps}
						/>
						<NumberInputStepper {...sliderNumberInputStepperProps}>
							<NumberIncrementStepper border="none" />
							<NumberDecrementStepper border="none" />
						</NumberInputStepper>
					</NumberInput>
				)}

				{withReset && (
					<IconButton
						size="sm"
						aria-label="Reset"
						tooltip="Reset"
						icon={<BiReset />}
						onClick={handleResetDisable}
						isDisabled={isResetDisabled}
						{...sliderIconButtonProps}
					/>
				)}
			</HStack>
		</FormControl>
	);
};
