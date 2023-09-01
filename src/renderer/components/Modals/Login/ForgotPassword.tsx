import React, { useState } from 'react'
import {
  Flex,
  Heading,
  Input,
  IconButton,
  Button,
  InputGroup,
  Stack,
  InputLeftElement,
  Box,
  Link,
  FormControl,
  Image,
} from '@chakra-ui/react'
import { IoIosMail } from 'react-icons/io'
import Logo from '../../../images/ArtroomLogo.png'
import axios from 'axios'
import { useRecoilState } from 'recoil'
import * as atom from '../../../atoms/atoms'

const ForgotPassword = ({
  setState,
}: {
  setState: React.Dispatch<React.SetStateAction<string>>
}) => {
  const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL

  const [email, setEmail] = useRecoilState(atom.emailState)
  function handleForgotPassword() {
    axios
      .get(`${ARTROOM_URL}/forgot_password_send_code`, {
        params: { email },
        headers: {
          'Content-Type': 'application/json',
          accept: 'application/json',
        },
      })
      .then((result) => {
        console.log(result)
        setState('ForgotPasswordCode')
      })
      .catch((err) => {
        console.log(err)
      })
  }

  return (
    <Flex
      alignItems="center"
      flexDirection="column"
      height="60vh"
      justifyContent="center"
      width="100wh"
    >
      <Stack alignItems="center" flexDir="column" justifyContent="center" mb="2">
        <Image h="50px" src={Logo} />

        <Heading color="blue.600">Welcome</Heading>

        <Box minW={{ base: '90%', md: '468px' }}>
          <Stack backgroundColor="whiteAlpha.900" boxShadow="md" p="1rem" spacing={4}>
            <FormControl>
              <InputGroup>
                <InputLeftElement
                  children={<IoIosMail color="gray.800" />}
                  color="gray.800"
                  pointerEvents="none"
                />

                <Input
                  _placeholder={{ color: 'gray.400' }}
                  borderColor="#00000020"
                  color="gray.800"
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="Email Address"
                  type="email"
                  value={email}
                />
              </InputGroup>
            </FormControl>
            <Button
              borderRadius={10}
              colorScheme="blue"
              onClick={handleForgotPassword}
              type="submit"
              variant="solid"
              width="full"
            >
              Reset Password
            </Button>
          </Stack>
        </Box>
      </Stack>

      <Box>
        New to us?{' '}
        <Link
          color="teal.500"
          href="#"
          onClick={() => {
            setState('SignUp')
          }}
        >
          Sign Up
        </Link>
      </Box>
    </Flex>
  )
}

export default ForgotPassword
