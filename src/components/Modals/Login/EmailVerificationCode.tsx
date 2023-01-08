import React, { useState } from 'react';
import {
    Flex,
    Heading,
    Button,
    Stack,
    Box,
    Link,
    Image,
    Text,
    createStandaloneToast 
} from '@chakra-ui/react';
import Logo from '../../../images/ArtroomLogo.png';
import axios from 'axios';
import PinInputCode from '../../Reusable/PinInput/PinInputCode';
import { useRecoilState } from 'recoil';
import * as atom from '../../../atoms/atoms'

const EmailVerificationCode = ({ setLoggedIn, setState }: { setLoggedIn: React.Dispatch<React.SetStateAction<boolean>>, setState: React.Dispatch<React.SetStateAction<string>> }) => {
    const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL;
    const [verificationCode, setVerificationCode] = useState('');
    const [email, setEmail] = useRecoilState(atom.emailState);
    const { ToastContainer, toast } = createStandaloneToast();

    function handleResendCode(){
        axios.get(
            `${ARTROOM_URL}/resend_code`,
            {
                params : {
                    email,
                },
                headers: {
                    'Content-Type': 'application/json',
                    'accept': 'application/json'
                }
            }
        ).then((result) => {
            console.log(result);
            toast({
                title: 'Sent Verification Code',
                description: "Please check your email (it may have went to spam)",
                status: 'success',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        });  
    }

    function handleAccountVerify() {
        let body = {
            email,
            code: verificationCode
        }
        console.log(body);
        axios.post(
            `${ARTROOM_URL}/verify_user`,
            body,
            {
            headers: {
                'Content-Type': 'application/json',
                'accept': 'application/json'
            }
            }
        ).then((result) => {
            console.log(result);
            setLoggedIn(true);
        }).catch(err => {
            console.log(err);
            toast({
                title: 'Verification Failed',
                description: "Error, Verification Failed",
                status: 'error',
                position: 'top',
                duration: 2000,
                isClosable: false,
                containerStyle: {
                    pointerEvents: 'none'
                }
            });
        });
    }

    return (
        <Flex
            alignItems="center"
            flexDirection="column"
            height="60vh"
            justifyContent="center"
            width="100wh"
        >
            <Stack
                alignItems="center"
                flexDir="column"
                justifyContent="center"
                mb="2"
            >
                <Image
                    h="50px"
                    src={Logo} />

                <Heading color="blue.600">
                    Welcome
                </Heading>

                <Box minW={{ base: '90%',
                    md: '468px' }}>
                    <Stack
                        backgroundColor="whiteAlpha.900"
                        boxShadow="md"
                        p="1rem"
                        spacing={4}
                    >
                        <Text width="500px" color="gray.800">
                            A verification code has been sent to your email. Please enter the code below. It will expire in 10 minutes. Please also check your spam.
                        </Text>
                        <PinInputCode verificationCode={verificationCode} setVerificationCode={setVerificationCode} numInputs={6} handleResendCode={handleResendCode}></PinInputCode>
                        <Button
                            borderRadius={10}
                            colorScheme="blue"
                            onClick={handleAccountVerify}
                            type="submit"
                            variant="solid"
                            width="full"
                        >
                            Verify
                        </Button>
                    </Stack>
                </Box>
            </Stack>

            <Box>
                New to us?
                {' '}

                <Link
                    color="teal.500"
                    href="#"
                    onClick={() => {
                        setState('SignUp');
                    }}>
                    Sign Up
                </Link>
            </Box>
        </Flex>

    );
};

export default EmailVerificationCode;
