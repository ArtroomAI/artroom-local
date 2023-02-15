import React from 'react'
import {
  Button as ChakraButton,
  ButtonProps,
  forwardRef,
  Tooltip,
  TooltipProps
} from '@chakra-ui/react'
import { ReactNode } from 'react'

export interface IButtonProps extends ButtonProps {
  tooltip?: string
  tooltipProps?: Omit<TooltipProps, 'children'>
  styleClass?: string
  children: ReactNode
}

export const Button = forwardRef((props: IButtonProps, forwardedRef) => {
  const { children, tooltip = '', tooltipProps, styleClass, ...rest } = props
  return (
    <Tooltip label={tooltip} {...tooltipProps}>
      <ChakraButton
        ref={forwardedRef}
        className={['painter__button', styleClass].join(' ')}
        {...rest}
      >
        {children}
      </ChakraButton>
    </Tooltip>
  )
})
