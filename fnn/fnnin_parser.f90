module mod

  type model_params
    character(len=:),allocatable :: element
    real(4) :: mass
    integer(4),allocatable :: hlayers(:)
  end type

  type fnn_param
    real(4),allocatable,dimension(:) :: rad_eta, rad_mu
    real(4),allocatable,dimension(:) :: ang_mu, ang_eta, ang_zeta
    integer(4),allocatable,dimension(:) :: ang_lambda

    real(4) :: rad_rc, rad_damp, ang_rc, ang_damp

    type(model_params), allocatable :: models(:) 

    contains 
       procedure :: print => fnn_param_print

  end type

   interface get_tokens_and_append
      module procedure :: get_tokens_and_append_rv, get_tokens_and_append_iv, get_tokens_and_append_rs, get_tokens_and_append_model
   end interface

   character(len=:),allocatable,private :: sbuf
   integer,private :: ibuf
   real(4),private :: rbuf
    character(len=:),allocatable,private :: token

contains

  subroutine get_tokens_and_append_model(linein, models)
    implicit none
    character(len=:),allocatable,intent(in out) :: linein
    type(model_params),allocatable,intent(in out) :: models(:)
    type(model_params) :: mbuf

    if (getstr(linein, token) < 0) stop 'erro while reading element name'  
    mbuf%element = trim(adjustl(token))
    if (getstr(linein, token) < 0) stop 'erro while reading element mass'  
    read(token, *) mbuf%mass

    ! allocate zero-sized array
    if(.not.allocated(mbuf%hlayers)) allocate(mbuf%hlayers(0))
    do while( getstr(linein, token) > 0 )
       read(token, *) ibuf
       mbuf%hlayers = [mbuf%hlayers, ibuf]
    enddo

    ! allocate zero-sized array
    if(.not.allocated(models)) allocate(models(0)) 
    models = [models, mbuf]

    return
  end subroutine

  subroutine get_tokens_and_append_rv(linein, array)
    implicit none
    character(len=:),allocatable,intent(in out) :: linein
    real(4),allocatable,intent(in out) :: array(:)

    ! allocate zero-sized array
    if(.not.allocated(array)) allocate(array(0)) 

    do while (getstr(linein, token) > 0) 
       read(token,*) rbuf 
       array = [array, rbuf]
    enddo

    return
  end subroutine

  subroutine get_tokens_and_append_iv(linein, array)
    implicit none
    character(len=:),allocatable,intent(in out) :: linein
    integer(4),allocatable,intent(in out) :: array(:)

    ! allocate zero-sized array
    if(.not.allocated(array)) allocate(array(0)) 

    do while (getstr(linein, token) > 0) 
       read(token,*) ibuf 
       array = [array, ibuf]
    enddo

    return
  end subroutine

  subroutine get_tokens_and_append_rs(linein, scalar)
    implicit none
    character(len=:),allocatable,intent(in out) :: linein
    real(4),intent(in out) :: scalar

    if (getstr(linein, token) > 0) then
       read(token,*) rbuf 
       scalar = rbuf
    endif 

    return
  end subroutine

  function fnn_param_ctor(path) result(c)
    implicit none
    character(len=:),allocatable,intent(in) :: path
    character(256) :: linein0
    character(len=:),allocatable :: linein

    type(fnn_param) :: c 
    integer :: iunit

    open(newunit=iunit, file=path, status='old', form='formatted')

    do while (.true.)
      read(iunit,'(a)',end=10) linein0
      linein = trim(adjustl(linein0))

      if(getstr(linein, token) > 0) then

        select case (token)
           case ('rad_eta')
             call get_tokens_and_append(linein, c%rad_eta)
           case ('rad_mu')
             call get_tokens_and_append(linein, c%rad_mu)
           case ('ang_eta')
             call get_tokens_and_append(linein, c%ang_eta)
           case ('ang_mu')
             call get_tokens_and_append(linein, c%ang_mu)
           case ('ang_lambda')
             call get_tokens_and_append(linein, c%ang_lambda)
           case ('ang_zeta')
             call get_tokens_and_append(linein, c%ang_zeta)

           case ('rad_rc')
             call get_tokens_and_append(linein, c%rad_rc)
           case ('rad_damp')
             call get_tokens_and_append(linein, c%rad_damp)
           case ('ang_rc')
             call get_tokens_and_append(linein, c%ang_rc)
           case ('ang_damp')
             call get_tokens_and_append(linein, c%ang_damp)
           case ('model')
             call get_tokens_and_append(linein, c%models)

           case default
        end select
     endif

    end do


    10 close(iunit)

  end function

  subroutine fnn_param_print(this)
    implicit none
    class(fnn_param), intent(in) :: this
    integer :: i,j

    print'(a)',repeat('-',60)
       print'(a,20f6.2)', 'rad_eta: ', this%rad_eta
       print'(a,20f6.2)', 'rad_mu: ', this%rad_mu
    print'(a)',repeat('-',60)
       print'(a,20f6.2)', 'ang_eta: ', this%ang_eta
       print'(a,20f6.2)', 'ang_mu: ', this%ang_mu
       print'(a,20i6)', 'ang_lambda: ', this%ang_lambda
       print'(a,20f6.2)', 'ang_zeta: ', this%ang_zeta
    print'(a)',repeat('-',60)
       print'(a,20f6.2)', 'rad_rc: ', this%rad_rc
       print'(a,20f6.2)', 'rad_damp: ', this%rad_damp
       print'(a,20f6.2)', 'ang_rc: ', this%ang_rc
       print'(a,20f6.2)', 'ang_damp: ', this%ang_damp
    print'(a)',repeat('-',60)

    do i = 1, size(this%models)
       associate(m => this%models(i))
          print*, m%element, m%mass, m%hlayers(:)
       end associate
    end do
    print'(a)',repeat('-',60)
    
  end subroutine

!-------------------------------------------------------------------------------------------
integer function getstr(linein,lineout)
implicit none
!-------------------------------------------------------------------------------------------

character(len=:),allocatable,intent(in out) :: linein,lineout
integer :: pos1

!--- remove whitespace 
linein = adjustl(linein)

!--- return if black line
if(len(linein)==0) then
  getstr=-2
  return
endif

!--- return if it's a comment line or entirely whitespace
if(linein(1:1)=='#' .or. linein == repeat(' ', len(linein)) ) then
   getstr=-1
   return
endif

! find position in linein to get a token. if whitespace is not found, take entire line
pos1=index(linein,' ')-1
if(pos1==-1) pos1=len(linein)

lineout=linein(1:pos1)
linein=linein(pos1+1:)
getstr=len(lineout)

return
end

!-------------------------------------------------------------------------------------------
logical function find_cmdline_argc(key,idx)
implicit none
!-------------------------------------------------------------------------------------------
integer,intent(inout) :: idx
character(*) :: key

integer,parameter :: MAXSTRLENGTH = 256

integer :: i
character(MAXSTRLENGTH) :: argv

do i=1, command_argument_count()
   call get_command_argument(i,argv)
   if(index(argv,trim(adjustl(key))//' ')/=0) then ! trailing zero to distinguish '-foo ' and '-fooo'
      idx=i
      find_cmdline_argc=.true.
      return
   endif
enddo

idx=-1
find_cmdline_argc=.false.

return
end function

end module

program main
  use mod
  implicit none
  integer :: unit
  type(fnn_param) :: f
  character(256) :: path0
  character(len=:),allocatable :: path
  path0 = './fnn.in'
  path = trim(adjustl(path0))

  f = fnn_param_ctor(path)
  call f%print()
  
end program
