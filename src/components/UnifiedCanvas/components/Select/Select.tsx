import React from 'react'
import {
  FormControl,
  FormLabel,
  Select as ChakraSelect,
  SelectProps,
  Tooltip,
  TooltipProps
} from '@chakra-ui/react'
import { MouseEvent, FC } from 'react'

type ISelectProps = SelectProps & {
  label?: string
  styleClass?: string
  tooltip?: string
  tooltipProps?: Omit<TooltipProps, 'children'>
  validValues:
    | Array<number | string>
    | Array<{ key: string; value: string | number }>
}
/**
 * Customized Chakra FormControl + Select multi-part component.
 */
export const Select: FC<ISelectProps> = props => {
  const {
    label,
    isDisabled,
    validValues,
    tooltip,
    tooltipProps,
    size = 'sm',
    fontSize = 'md',
    styleClass,
    ...rest
  } = props
  return (
    <FormControl
      isDisabled={isDisabled}
      className={`painter__select ${styleClass}`}
      onClick={(e: MouseEvent<HTMLDivElement>) => {
        e.stopPropagation()
        e.nativeEvent.stopImmediatePropagation()
        e.nativeEvent.stopPropagation()
        e.nativeEvent.cancelBubble = true
      }}
    >
      {label && (
        <FormLabel
          className="painter__select-label"
          fontSize={fontSize}
          marginBottom={1}
          flexGrow={2}
          whiteSpace="nowrap"
        >
          {label}
        </FormLabel>
      )}
      <Tooltip label={tooltip} {...tooltipProps}>
        <ChakraSelect
          className="painter__select-picker"
          fontSize={fontSize}
          size={size}
          {...rest}
        >
          {validValues.map(opt => {
            return typeof opt === 'string' || typeof opt === 'number' ? (
              <option key={opt} value={opt} className="painter__select-option">
                {opt}
              </option>
            ) : (
              <option
                key={opt.value}
                value={opt.value}
                className="painter__select-option"
              >
                {opt.key}
              </option>
            )
          })}
        </ChakraSelect>
      </Tooltip>
    </FormControl>
  )
}
