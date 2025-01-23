#include <Windows.h>
#include <io.h>
#include <fcntl.h>

#include <iostream>

#include <cstdint>
#include <cstdlib>

#define CURRENT_WND_CLASS L"GameWndClass_Didiet"
#define DEF_CX            800
#define DEF_CY            600

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

HMODULE GetCurrentModuleHandle()
{
    HMODULE ImageBase;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&GetCurrentModuleHandle,
                           &ImageBase))
    {
        return ImageBase;
    }
    return 0;
}

#if 1
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
#else
int main()
{
    HINSTANCE hInstance = GetCurrentModuleHandle();
#endif
    std::wcout << L"HEllo Windwos Application\n";

    WNDCLASSEX wcex; /* Structure needed for creating Window */
    HWND       hWnd; /* Window Handle */
    MSG        msg;
    BOOL       bDone = FALSE;
    SIZE       screenSize;
    LONG       winX, winY;

    ZeroMemory(&wcex, sizeof(WNDCLASSEX));
    ZeroMemory(&msg, sizeof(MSG));

    wcex.cbSize        = sizeof(WNDCLASSEX);
    wcex.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wcex.hCursor       = LoadCursor(hInstance, IDC_ARROW);
    wcex.hIcon         = LoadIcon(hInstance, IDI_APPLICATION);
    wcex.hInstance     = hInstance;
    wcex.lpfnWndProc   = WndProc;
    wcex.lpszClassName = CURRENT_WND_CLASS;
    wcex.style         = CS_HREDRAW | CS_VREDRAW;
    wcex.cbWndExtra    = 0;
    wcex.cbClsExtra    = 0;
    wcex.lpszMenuName  = NULL;

    if (!RegisterClassEx(&wcex))
    {
        return -1;
    }

    screenSize.cx = GetSystemMetrics(SM_CXSCREEN);
    screenSize.cy = GetSystemMetrics(SM_CYSCREEN);

    winX = (screenSize.cx - (DEF_CX + GetSystemMetrics(SM_CXBORDER) * 2)) / 2;
    winY = (screenSize.cy - (DEF_CY + GetSystemMetrics(SM_CYBORDER) + GetSystemMetrics(SM_CYCAPTION))) / 2;

    hWnd = CreateWindowEx(WS_EX_OVERLAPPEDWINDOW,
                          CURRENT_WND_CLASS,
                          L"Game Window",
                          WS_OVERLAPPEDWINDOW,
                          winX,
                          winY,
                          DEF_CX,
                          DEF_CY,
                          0,
                          0,
                          hInstance,
                          0);

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    while (FALSE == bDone)
    {
        if (PeekMessage(&msg, hWnd, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            if (msg.message == WM_QUIT)
            {
                bDone = TRUE;
            }
        }
        else
        {
            /* do rendering here */
        }
    }

    DestroyWindow(hWnd);
    UnregisterClass(CURRENT_WND_CLASS, hInstance);
    return 0;

    std::wcout << L"Press Anything to exit..." << std::endl;
    std::cin.get();
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
        case WM_QUIT:
        case WM_DESTROY:
        case WM_CLOSE:
            std::wcout << "EXITING" << std::endl;
            PostQuitMessage(0);
            break;
        default: return DefWindowProc(hWnd, uMsg, wParam, lParam);
    }
    return 0;
}
