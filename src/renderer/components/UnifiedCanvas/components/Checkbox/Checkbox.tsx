import React from 'react'
import { FC } from 'react'
import { Checkbox as ChakraCheckbox, CheckboxProps } from '@chakra-ui/react'

type ICheckboxProps = CheckboxProps & {
  label: string
  styleClass?: string
}

export const Checkbox: FC<ICheckboxProps> = (props) => {
  const { label, styleClass, ...rest } = props
  return (
    <ChakraCheckbox className={`painter__checkbox ${styleClass}`} {...rest}>
      {label}
    </ChakraCheckbox>
  )
}
