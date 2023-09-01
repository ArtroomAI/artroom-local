import React from 'react'
import { Flex, Text, Icon, Link, Menu, MenuButton, HStack, Tooltip, Image } from '@chakra-ui/react'
import type { IconType } from 'react-icons'

interface NavItemProps {
  icon: IconType | string
  title: string
  active?: boolean
  navSize: string
  linkTo?: string
  className: string
  onClick?: () => void
}

export default function NavItem({
  icon,
  title,
  active,
  navSize,
  linkTo,
  className,
  onClick,
}: NavItemProps) {
  return (
    <Flex
      alignItems={navSize === 'small' ? 'center' : 'flex-start'}
      flexDir="column"
      fontSize="md"
      mt={15}
      w="100%"
      onClick={onClick}
    >
      <Menu placement="right">
        <Link
          _hover={{ textDecor: 'none', backgroundColor: '#AEC8CA' }}
          borderRadius={8}
          href={linkTo}
          p={2.5}
          width="100%"
          textAlign="center"
          height="46px"
        >
          <Tooltip
            fontSize="md"
            label={navSize === 'small' ? title : ''}
            placement="bottom"
            shouldWrapChildren
          >
            <MenuButton bg="transparent" width="100%">
              <HStack className={className}>
                {typeof icon === 'string' ? (
                  <Image
                    color="#82AAAD"
                    fontSize="xl"
                    justifyContent="center"
                    src={icon}
                    width="20px"
                    height="20px"
                  />
                ) : (
                  <Icon
                    as={icon}
                    color={active ? '#82AAAD' : 'gray.500'}
                    fontSize="xl"
                    justifyContent="center"
                  />
                )}

                <Text
                  align="center"
                  display={navSize === 'small' ? 'none' : 'flex'}
                  fontSize="m"
                  pl={5}
                  pr={10}
                >
                  {title}
                </Text>
              </HStack>
            </MenuButton>
          </Tooltip>
        </Link>
      </Menu>
    </Flex>
  )
}
