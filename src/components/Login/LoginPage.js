import React from 'react';
import { useState } from 'react';
import {
    useDisclosure,
    Button,
    Modal,
    ModalOverlay,
    ModalContent,
    ModalCloseButton
} from '@chakra-ui/react';
import Login from './Login';
import SignUp from './Signup';
import ForgotPassword from './ForgotPassword';
import ForgotPasswordCode from './ForgotPasswordCode';
import EmailVerificationCode from './EmailVerificationCode';
import ResetPassword from './ResetPassword';

const LoginPage = ({ setLoggedIn }) => {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const [state, setState] = useState('Login');
    const [pwdResetJwt, setPwdResetJwt] = useState('');

    return (
        <>
            <Button
                aria-label="View"
                onClick={onOpen}
                varient="outline">
                Login
                {' '}
            </Button>

            <Modal
                isOpen={isOpen}
                motionPreset="slideInBottom"
                onClose={onClose}
                scrollBehavior="outside"
                size="4xl">
                <ModalOverlay />

                <ModalContent bg="gray.900">
                    <ModalCloseButton />
                    {state === 'Login' ?
                        <Login setLoggedIn={setLoggedIn} setState={setState}/>
                    : state === 'SignUp' ? 
                        <SignUp setState={setState} />
                    : state === 'ForgotPassword' ? 
                        <ForgotPassword setState={setState}></ForgotPassword>
                    : state === 'ForgotPasswordCode' ? 
                        <ForgotPasswordCode setState={setState} setPwdResetJwt={setPwdResetJwt} ></ForgotPasswordCode>
                    : state === 'EmailVerificationCode' ?
                        <EmailVerificationCode setLoggedIn={setLoggedIn} setState={setState}></EmailVerificationCode>
                    : state === 'ResetPassword' ?
                        <ResetPassword setState={setState} pwdResetJwt={pwdResetJwt} ></ResetPassword>
                    : <></>
                    }
                </ModalContent>
            </Modal>
        </>

    );
};

export default LoginPage;
