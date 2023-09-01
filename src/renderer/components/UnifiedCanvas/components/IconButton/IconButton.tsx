import React from 'react'
import {
  IconButtonProps,
  IconButton as ChakraIconButton,
  Tooltip,
  TooltipProps,
  forwardRef,
} from '@chakra-ui/react'

export type IIconButtonProps = IconButtonProps & {
  styleClass?: string
  tooltip?: string
  tooltipProps?: Omit<TooltipProps, 'children'>
  asCheckbox?: boolean
  isChecked?: boolean
}

export const IconButton = forwardRef((props: IIconButtonProps, forwardedRef) => {
  const { tooltip = '', styleClass, tooltipProps, asCheckbox, isChecked, ...rest } = props

  return (
    <Tooltip
      label={tooltip}
      hasArrow
      {...tooltipProps}
      {...(tooltipProps?.placement ? { placement: tooltipProps.placement } : { placement: 'top' })}
    >
      <ChakraIconButton
        ref={forwardedRef}
        className={styleClass ? `painter__icon-button ${styleClass}` : `painter__icon-button`}
        data-as-checkbox={asCheckbox}
        data-selected={isChecked !== undefined ? isChecked : undefined}
        {...rest}
      />
    </Tooltip>
  )
})
