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
  styleClass?: string;
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

export function Slider(props: IFullSliderProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const {
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
    styleClass,
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
  } = props;

  const [localInputValue, setLocalInputValue] = useState<string>(String(value));

  const numberInputMax = useMemo(
    () => (sliderNumberInputProps?.max ? sliderNumberInputProps.max : max),
    [max, sliderNumberInputProps?.max]
  );

  useEffect(() => {
    if (String(value) !== localInputValue && localInputValue !== '') {
      setLocalInputValue(String(value));
    }
  }, [value, localInputValue, setLocalInputValue]);

  const handleInputBlur = (e: FocusEvent<HTMLInputElement>) => {
    const clamped = _.clamp(
      isInteger ? Math.floor(Number(e.target.value)) : Number(e.target.value),
      min,
      numberInputMax
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
      className={
        styleClass
          ? `painter__slider-component ${styleClass}`
          : `painter__slider-component`
      }
      data-markers={withSliderMarks}
      {...sliderFormControlProps}
    >
      <FormLabel
        className="painter__slider-component-label"
        {...sliderFormLabelProps}
      >
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
          {...rest}
        >
          {withSliderMarks && (
            <>
              <SliderMark
                value={min}
                className="painter__slider-mark painter__slider-mark-start"
                ml={sliderMarkLeftOffset}
                {...sliderMarkProps}
              >
                {min}
              </SliderMark>
              <SliderMark
                value={max}
                className="painter__slider-mark painter__slider-mark-end"
                ml={sliderMarkRightOffset}
                {...sliderMarkProps}
              >
                {max}
              </SliderMark>
            </>
          )}

          <SliderTrack className="painter__slider_track" {...sliderTrackProps}>
            <SliderFilledTrack className="painter__slider_track-filled" />
          </SliderTrack>

          <Tooltip
            hasArrow
            className="painter__slider-component-tooltip"
            placement="top"
            isOpen={showTooltip}
            label={`${value}${tooltipSuffix}`}
            hidden={hideTooltip}
            {...sliderTooltipProps}
          >
            <SliderThumb
              className="painter__slider-thumb"
              {...sliderThumbProps}
            />
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
            className="painter__slider-number-field"
            isDisabled={isInputDisabled}
            {...sliderNumberInputProps}
          >
            <NumberInputField
              className="painter__slider-number-input"
              width={inputWidth}
              readOnly={inputReadOnly}
              {...sliderNumberInputFieldProps}
            />
            <NumberInputStepper {...sliderNumberInputStepperProps}>
              <NumberIncrementStepper className="painter__slider-number-stepper" />
              <NumberDecrementStepper className="painter__slider-number-stepper" />
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
}
